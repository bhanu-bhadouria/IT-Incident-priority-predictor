from predict import predict_priority

# Case 1: Clearly High Priority
label, conf, prob, _ = predict_priority(
    impact=1, urgency=1,
    reassignment_count=4, reopen_count=2,
    contact_type="Phone",
    category="Network", subcategory="Connectivity",
    opened_at="2024-01-15 02:30:00",   # 2am Monday — night incident
    sys_mod_count=8,
    notify=1,
)
print("=== Case 1: Clearly High Priority ===")
print(f"Prediction  : {label}")
print(f"Confidence  : {conf:.2f}%")
print(f"Would alert on-call: {'Yes' if label == 'High Priority' else 'No'}")
print()

# Case 2: Clearly Normal  — impact=3, urgency=3, business hours, no reassignments
label, conf, prob, _ = predict_priority(
    impact=3, urgency=3,
    reassignment_count=0, reopen_count=0,
    contact_type="Email",
    category="Software", subcategory="Installation",
    opened_at="2024-01-15 10:30:00",   # 10:30am Monday — business hours
    sys_mod_count=0,
    notify=0,
)
print("=== Case 2: Clearly Normal ===")
print(f"Prediction  : {label}")
print(f"Confidence  : {conf:.2f}%")
print(f"Would alert on-call: {'Yes' if label == 'High Priority' else 'No'}")
print()

# Case 3: Ambiguous       — impact=2, urgency=2, weekend, one reassignment
label, conf, prob, _ = predict_priority(
    impact=2, urgency=2,
    reassignment_count=1, reopen_count=0,
    contact_type="Email",
    category="Software", subcategory="Installation",
    opened_at="2024-01-15 18:30:00",   # 6:30pm Monday — weekend
    sys_mod_count=1,
    notify=0,
)
print("=== Case 3: Ambiguous ===")
print(f"Prediction  : {label}")
print(f"Confidence  : {conf:.2f}%")
print(f"Would alert on-call: {'Yes' if label == 'High Priority' else 'No'}")