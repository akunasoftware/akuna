use super::models::pp_doclayout::{LayoutDetection, sort_detections_by_order};

#[test]
fn order_ties_are_stable() {
    let detections = vec![
        LayoutDetection {
            label: "z".to_string(),
            score: 0.5,
            bbox: [10.0, 20.0, 30.0, 40.0],
            order: 1,
        },
        LayoutDetection {
            label: "a".to_string(),
            score: 0.5,
            bbox: [10.0, 20.0, 30.0, 40.0],
            order: 1,
        },
    ];

    let sorted = sort_detections_by_order(detections);

    assert_eq!(sorted[0].label, "a");
    assert_eq!(sorted[1].label, "z");
}
