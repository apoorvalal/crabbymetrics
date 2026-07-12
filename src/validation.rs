use ndarray::{Array1, ArrayBase, Data, Dimension};

pub(crate) fn validate_finite<S, D>(name: &str, values: &ArrayBase<S, D>) -> Result<(), String>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

pub(crate) fn validate_nonnegative<S, D>(name: &str, values: &ArrayBase<S, D>) -> Result<(), String>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(format!(
            "{name} must contain only finite nonnegative values"
        ));
    }
    Ok(())
}

pub(crate) fn validate_positive<S, D>(name: &str, values: &ArrayBase<S, D>) -> Result<(), String>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(format!("{name} must contain only positive finite values"));
    }
    Ok(())
}

pub(crate) fn validate_binary_f64(name: &str, values: &Array1<f64>) -> Result<(), String> {
    if values
        .iter()
        .any(|value| !value.is_finite() || (*value != 0.0 && *value != 1.0))
    {
        return Err(format!("{name} must contain only 0 and 1"));
    }
    Ok(())
}

pub(crate) fn validate_weights(name: &str, weights: &Array1<f64>, n: usize) -> Result<(), String> {
    if weights.len() != n {
        return Err(format!(
            "{name} length must match the number of observations"
        ));
    }
    if weights
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(format!("{name} values must be finite and nonnegative"));
    }
    if weights.iter().all(|value| *value == 0.0) {
        return Err(format!("{name} must contain at least one positive value"));
    }
    Ok(())
}

pub(crate) fn validate_sample_weight(weights: &Array1<f64>, n: usize) -> Result<(), String> {
    validate_weights("sample_weight", weights, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn sample_weights_need_positive_mass() {
        assert!(validate_sample_weight(&array![0.0, 0.0], 2).is_err());
        assert!(validate_sample_weight(&array![0.0, 1.0], 2).is_ok());
    }
}
