WOLISZ_SCHEDULE = {
    "Monday": [
        (0 * 3600, 7 * 3600, "night"),
        (7 * 3600, 9 * 3600, "active"),
        (9 * 3600, 18 * 3600, "not_present"),
        (18 * 3600, 23 * 3600, "inactive"),
        (23 * 3600, 24 * 3600, "night")
    ],
    "Tuesday": [
        (0 * 3600, 7 * 3600, "night"),
        (7 * 3600, 9 * 3600, "active"),
        (9 * 3600, 18 * 3600, "not_present"),
        (18 * 3600, 23 * 3600, "inactive"),
        (23 * 3600, 24 * 3600, "night")
    ],
    "Thursday": [
        (0 * 3600, 7 * 3600, "night"), #295/292
        (7 * 3600, 9 * 3600, "active"), #296/293
        (9 * 3600, 18 * 3600, "not_present"), #297/292
        (18 * 3600, 23 * 3600, "inactive"), #298/295
        (23 * 3600, 24 * 3600, "night") #295/292
    ],
    "Friday": [
        (0 * 3600, 7 * 3600, "night"),
        (7 * 3600, 9 * 3600, "active"),
        (9 * 3600, 18 * 3600, "not_present"),
        (18 * 3600, 23 * 3600, "inactive"),
        (23 * 3600, 24 * 3600, "night")
    ],
    "Saturday": [
        (0 * 3600, 8 * 3600, "night"),
        (8 * 3600, 18 * 3600, "active"),
        (18 * 3600, 23 * 3600, "inactive"),
        (23 * 3600, 24 * 3600, "night"),
    ],
    "Sunday": [
        (0 * 3600, 6 * 3600, "night"), #295/292
        (6 * 3600, 12 * 3600, "active"), #296/293
        (12 * 3600, 18 * 3600, "night"), #295/292
        (18 * 3600, 24 * 3600, "active"), #296/293
    ]
}

SIMPLE_SCHEDULE = {
    "Monday": [
        (0 * 3600, 8 * 3600, "night"),
        (8 * 3600, 16 * 3600, "active"),
        (16 * 3600, 24 * 3600, "inactive"),
    ],
    "Tuesday": [
        (0 * 3600, 8 * 3600, "night"),
        (8 * 3600, 16 * 3600, "active"),
        (16 * 3600, 24 * 3600, "inactive"),
    ],
    "Wednesday": [
        (0 * 3600, 8 * 3600, "night"),
        (8 * 3600, 16 * 3600, "active"),
        (16 * 3600, 24 * 3600, "inactive"),
    ],
    "Thursday": [
        (0 * 3600, 8 * 3600, "night"),
        (8 * 3600, 16 * 3600, "active"),
        (16 * 3600, 24 * 3600, "inactive"),
    ],
    "Friday": [
        (0 * 3600, 8 * 3600, "night"),
        (8 * 3600, 16 * 3600, "active"),
        (16 * 3600, 24 * 3600, "inactive"),
    ],
    "Saturday": [
        (0 * 3600, 8 * 3600, "night"),
        (8 * 3600, 16 * 3600, "active"),
        (16 * 3600, 24 * 3600, "inactive"),
    ],
    "Sunday": [
        (0 * 3600, 8 * 3600, "night"),
        (8 * 3600, 16 * 3600, "active"),
        (16 * 3600, 24 * 3600, "inactive"),
    ],
}

TEST_SCHEDULE = {
    "Monday": [
        (0 * 3600, 24 * 3600, "night"),
    ],
    "Tuesday": [
        (0 * 3600, 24 * 3600, "night"),
    ],
    "Wednesday": [
        (0 * 3600, 24 * 3600, "night"),
    ],
    "Thursday": [
        (0 * 3600, 24 * 3600, "night"),
    ],
    "Friday": [
        (0 * 3600, 24 * 3600, "night"),
    ],
    "Saturday": [
        (0 * 3600, 24 * 3600, "night"),
    ],
    "Sunday": [
        (0 * 3600, 24 * 3600, "night"),
    ],
}
SCHEDULES = {
    "wolisz": WOLISZ_SCHEDULE,
    "simple": SIMPLE_SCHEDULE,
    "test": TEST_SCHEDULE,
}