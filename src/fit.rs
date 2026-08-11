use argmin::core::{TerminationReason, TerminationStatus};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Clone, Debug)]
pub(crate) struct FitDiagnostics {
    pub(crate) converged: bool,
    pub(crate) iterations: u64,
    pub(crate) termination_reason: String,
    pub(crate) objective: Option<f64>,
}

impl FitDiagnostics {
    pub(crate) fn new(
        converged: bool,
        iterations: u64,
        termination_reason: impl Into<String>,
        objective: Option<f64>,
    ) -> Self {
        Self {
            converged,
            iterations,
            termination_reason: termination_reason.into(),
            objective,
        }
    }

    pub(crate) fn from_argmin(
        status: &TerminationStatus,
        iterations: u64,
        objective: Option<f64>,
    ) -> Self {
        Self::new(
            optimization_success(status),
            iterations,
            status.to_string(),
            objective,
        )
    }

    pub(crate) fn require_converged(&self, estimator: &str) -> Result<(), String> {
        if self.converged {
            Ok(())
        } else {
            Err(format!(
                "{estimator} optimization did not converge after {} iterations: {}",
                self.iterations, self.termination_reason
            ))
        }
    }

    pub(crate) fn write_status(&self, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        dict.set_item("converged", self.converged)?;
        dict.set_item("iterations", self.iterations)?;
        dict.set_item("termination_reason", &self.termination_reason)?;
        Ok(())
    }

    pub(crate) fn write_summary(&self, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        self.write_status(dict)?;
        dict.set_item("objective", self.objective)?;
        Ok(())
    }
}

pub(crate) fn optimization_success(status: &TerminationStatus) -> bool {
    matches!(
        status,
        TerminationStatus::Terminated(TerminationReason::SolverConverged)
            | TerminationStatus::Terminated(TerminationReason::TargetCostReached)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iteration_budget_is_not_success() {
        let status = TerminationStatus::Terminated(TerminationReason::MaxItersReached);
        let diagnostics = FitDiagnostics::from_argmin(&status, 10, Some(1.0));
        assert!(!diagnostics.converged);
        assert!(diagnostics.require_converged("test solver").is_err());
    }
}
