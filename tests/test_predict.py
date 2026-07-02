from src.predict import predict_priority
import numpy as np

# pytest looks for functions starting with test_
def test_high_priority_prediction():
    label, conf, prob, _ = predict_priority(
        impact=1,
        urgency=1,
        reassignment_count=4,
        reopen_count=2,
        contact_type="Phone",
        category="Network",
        subcategory="Connectivity",
        opened_at="2024-01-15 02:30:00",
        sys_mod_count=8,
        notify=1,
    )
    assert label == "High Priority"      # assertion — test passes or fails here
    assert prob > 0.7                    # probability above threshold
    assert isinstance(conf, (float, np.floating))  # accepts both     

def test_normal_prediction():
    label, conf, prob, _ = predict_priority(
        impact=3,
        urgency=3,
        reassignment_count=0,
        reopen_count=0,
        contact_type="Email",
        category="Software",
        subcategory="Installation",
        opened_at="2024-01-15 10:30:00",
        sys_mod_count=0,
        notify=0,
    )
    assert label == "Normal"
    assert prob < 0.7