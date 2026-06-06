use std::sync::Arc;
use crate::config;
use crate::runtime::{Runtime, StreamWriter};
use crate::runnable::RunnableError;
use super::PregelExecutableTask;

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
        Self { runtime, stream_writer: None }
    }

    pub fn with_stream_writer(mut self, writer: StreamWriter) -> Self {
        self.stream_writer = Some(writer);
        self
    }

    /// Execute tasks in parallel (async).
    ///
    /// Each task's runnable is invoked with its input and config.
    /// Writes are collected into each task's write buffer.
    pub async fn run_tasks(&self, tasks: &mut [PregelExecutableTask]) -> Result<(), RunnerError> {
        if tasks.is_empty() {
            return Ok(());
        }

        if tasks.len() == 1 {
            let task = &mut tasks[0];
            Self::execute_single_task(task, self.runtime.as_ref(), self.stream_writer.clone()).await?;
            return Ok(());
        }

        for task in tasks.iter_mut() {
            Self::execute_single_task(task, self.runtime.as_ref(), self.stream_writer.clone()).await?;
        }

        Ok(())
    }

    /// Execute tasks synchronously (blocking).
    pub fn run_tasks_sync(&self, tasks: &mut [PregelExecutableTask]) -> Result<(), RunnerError> {
        for task in tasks.iter_mut() {
            Self::execute_single_task_sync(task, self.runtime.as_ref())?;
        }
        Ok(())
    }

    /// Execute a single task asynchronously.
    async fn execute_single_task(
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
