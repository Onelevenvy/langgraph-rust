use super::PregelExecutableTask;
use crate::config;
use crate::runnable::RunnableError;
use crate::runtime::{Runtime, StreamWriter};
use std::sync::Arc;
use tokio::task::JoinSet;

/// Dispatches tasks for parallel execution using tokio.
///
/// In the BSP model, all tasks in a super-step can run concurrently.
/// The runner dispatches them via `tokio::task::JoinSet` and collects
/// results as they complete.
pub struct PregelRunner {
    /// Optional runtime for config propagation.
    runtime: Option<Arc<Runtime>>,
    /// Optional stream writer for custom streaming.
    stream_writer: Option<StreamWriter>,
}

impl PregelRunner {
    pub fn new(runtime: Option<Arc<Runtime>>) -> Self {
        Self {
            runtime,
            stream_writer: None,
        }
    }

    pub fn with_stream_writer(mut self, writer: StreamWriter) -> Self {
        self.stream_writer = Some(writer);
        self
    }

    /// Execute tasks in parallel (async).
    ///
    /// Each task's runnable is invoked with its input and config.
    /// Writes are collected into each task's write buffer.
    ///
    /// A single task is executed inline to avoid spawn overhead on the common
    /// sequential-chain path; multiple tasks are dispatched through a
    /// `JoinSet` and run concurrently (fan-out takes ~max branch time instead
    /// of the sum of branch times).
    ///
    /// Returns the tasks (with their write buffers populated) alongside the
    /// overall result, since the caller inspects `task.writes` whether the
    /// step succeeded or was interrupted. If several tasks fail or interrupt,
    /// the lowest-index one wins, mirroring the order a serial runner would
    /// have reported them in and keeping the outcome deterministic regardless
    /// of `JoinSet` completion order. All tasks run to completion: a task that
    /// would come after a failing one in serial order still executes and keeps
    /// its writes (it ran in this super-step regardless).
    pub async fn run_tasks(
        &self,
        mut tasks: Vec<PregelExecutableTask>,
    ) -> (Vec<PregelExecutableTask>, Result<(), RunnerError>) {
        if tasks.is_empty() {
            return (tasks, Ok(()));
        }

        if tasks.len() == 1 {
            let task = &mut tasks[0];
            if let Err(e) =
                Self::execute_task(task, self.runtime.as_ref(), self.stream_writer.clone()).await
            {
                return (tasks, Err(e));
            }
            return (tasks, Ok(()));
        }

        let mut set = JoinSet::new();
        for (idx, mut task) in tasks.into_iter().enumerate() {
            let runtime = self.runtime.clone();
            let stream_writer = self.stream_writer.clone();
            set.spawn(async move {
                let result = Self::execute_task(&mut task, runtime.as_ref(), stream_writer).await;
                (idx, task, result)
            });
        }

        let mut done: Vec<(usize, PregelExecutableTask)> = Vec::with_capacity(set.len());
        let mut first_error: Option<(usize, RunnerError)> = None;

        while let Some(joined) = set.join_next().await {
            match joined {
                Ok((idx, task, result)) => {
                    if let Err(e) = result {
                        let replaces = match &first_error {
                            Some((i, _)) => idx < *i,
                            None => true,
                        };
                        if replaces {
                            first_error = Some((idx, e));
                        }
                    }
                    done.push((idx, task));
                }
                Err(join_err) => {
                    // A task whose future panicked. The task itself is lost
                    // (JoinSet cannot identify it); report a generic failure.
                    let msg = join_err
                        .try_into_panic()
                        .ok()
                        .and_then(|payload| {
                            payload.downcast_ref::<String>().cloned().or_else(|| {
                                payload.downcast_ref::<&str>().map(|s| (*s).to_string())
                            })
                        })
                        .unwrap_or_else(|| "task panicked".to_string());
                    if first_error.is_none() {
                        first_error = Some((
                            usize::MAX,
                            RunnerError::TaskFailed("<unknown>".to_string(), msg),
                        ));
                    }
                }
            }
        }

        // Restore serial order so streaming update emission is deterministic.
        done.sort_by_key(|(idx, _)| *idx);
        let tasks: Vec<PregelExecutableTask> = done.into_iter().map(|(_, t)| t).collect();

        match first_error {
            Some((_, e)) => (tasks, Err(e)),
            None => (tasks, Ok(())),
        }
    }

    /// Execute tasks synchronously (blocking).
    pub fn run_tasks_sync(&self, tasks: &mut [PregelExecutableTask]) -> Result<(), RunnerError> {
        for task in tasks.iter_mut() {
            Self::execute_single_task_sync(task, self.runtime.as_ref())?;
        }
        Ok(())
    }

    /// Execute a single task asynchronously.
    async fn execute_task(
        task: &mut PregelExecutableTask,
        runtime: Option<&Arc<Runtime>>,
        stream_writer: Option<StreamWriter>,
    ) -> Result<(), RunnerError> {
        let mut config = task.config.clone();
        {
            let configurable = config
                .entry("configurable".to_string())
                .or_insert_with(|| serde_json::json!({}));
            if let Some(obj) = configurable.as_object_mut() {
                obj.insert(
                    crate::constants::CONFIG_KEY_SEND.to_string(),
                    serde_json::json!(true),
                );
            }
        }

        // Build runtime with stream_writer if provided
        let effective_runtime = if let Some(rt) = runtime {
            if stream_writer.is_some() {
                let mut new_rt = (**rt).clone();
                new_rt.stream_writer = stream_writer;
                Some(Arc::new(new_rt))
            } else {
                Some(rt.clone())
            }
        } else if stream_writer.is_some() {
            Some(Arc::new(Runtime {
                context: (),
                store: None,
                stream_writer,
                previous: None,
                execution_info: None,
                server_info: None,
            }))
        } else {
            None
        };

        let result = if let Some(ref rt) = effective_runtime {
            config::with_runtime(config.clone(), rt.clone(), async {
                task.proc.ainvoke(&task.input, &config).await
            })
            .await
        } else {
            task.proc.ainvoke(&task.input, &config).await
        };

        match result {
            Ok(output) => {
                if let Some(obj) = output.as_object() {
                    for (key, val) in obj {
                        task.writes.push((key.clone(), val.clone()));
                    }
                }
            }
            Err(RunnableError::Interrupt(interrupt)) => {
                // Return the task_id along with the interrupt so the caller
                // can save the interrupt as a pending write in the checkpoint.
                return Err(RunnerError::Interrupt {
                    task_id: task.id.clone(),
                    interrupt,
                });
            }
            Err(e) => {
                return Err(RunnerError::TaskFailed(task.name.clone(), e.to_string()));
            }
        }

        Ok(())
    }

    /// Execute a single task synchronously.
    fn execute_single_task_sync(
        task: &mut PregelExecutableTask,
        runtime: Option<&Arc<Runtime>>,
    ) -> Result<(), RunnerError> {
        let mut config = task.config.clone();
        {
            let configurable = config
                .entry("configurable".to_string())
                .or_insert_with(|| serde_json::json!({}));
            if let Some(obj) = configurable.as_object_mut() {
                obj.insert(
                    crate::constants::CONFIG_KEY_SEND.to_string(),
                    serde_json::json!(true),
                );
            }
        }

        let result = if let Some(rt) = runtime {
            config::with_runtime_sync(config.clone(), rt.clone(), || {
                task.proc.invoke(&task.input, &config)
            })
        } else {
            task.proc.invoke(&task.input, &config)
        };

        match result {
            Ok(output) => {
                if let Some(obj) = output.as_object() {
                    for (key, val) in obj {
                        task.writes.push((key.clone(), val.clone()));
                    }
                }
            }
            Err(RunnableError::Interrupt(interrupt)) => {
                return Err(RunnerError::Interrupt {
                    task_id: task.id.clone(),
                    interrupt,
                });
            }
            Err(e) => {
                return Err(RunnerError::TaskFailed(task.name.clone(), e.to_string()));
            }
        }

        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    #[error("task '{0}' failed: {1}")]
    TaskFailed(String, String),

    #[error("graph interrupt")]
    Interrupt {
        task_id: String,
        interrupt: crate::types::GraphInterrupt,
    },
}
