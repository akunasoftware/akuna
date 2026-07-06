use super::*;

#[test]
fn util_batch_size_validate_rejects_zero() {
    let error = resolve_batch_size(1, Some(0), DEFAULT_BATCH_SIZE)
        .expect_err("zero batch size should fail");
    assert!(
        error
            .to_string()
            .contains("batch size must be greater than zero")
    );
}

#[test]
fn util_top_k_validate_rejects_zero() {
    let error = validate_top_k(Some(0)).expect_err("zero top_k should fail");
    assert!(
        error
            .to_string()
            .contains("top_k must be greater than zero")
    );
}

#[test]
fn util_top_k_validate_accepts_none_and_positive() {
    validate_top_k(None).expect("None top_k should pass");
    validate_top_k(Some(1)).expect("positive top_k should pass");
}

#[test]
fn util_sigmoid_maps_scores_to_zero_one() {
    assert_eq!(sigmoid_f32(0.0), 0.5);
    assert!(sigmoid_f32(10.0) > 0.99);
    assert!(sigmoid_f32(-10.0) < 0.01);
}

#[test]
fn util_sigmoid_bounded_for_extreme_scores() {
    assert!(sigmoid_f32(1000.0).is_finite());
    assert!(sigmoid_f32(-1000.0).is_finite());
    assert!(sigmoid_f32(1000.0) <= 1.0);
    assert!(sigmoid_f32(-1000.0) >= 0.0);
}
