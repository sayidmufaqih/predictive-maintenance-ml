FEATURE_RANGES = {
    "Air temperature [K]": (295.3, 304.5),
    "Process temperature [K]": (305.7, 313.8),
    "Rotational speed [rpm]": (1168, 2886),
    "Torque [Nm]": (3.8, 76.6),
    "Tool wear [min]": (0, 253)
}


def validate_machine_input(input_data):

    errors = []

    # Check training-data range
    for feature, (min_value, max_value) in FEATURE_RANGES.items():

        value = input_data[feature].iloc[0]

        if value < min_value or value > max_value:
            errors.append(
                f"{feature}: {value} is outside the "
                f"supported range ({min_value}–{max_value})."
            )

    # Logical validation
    air_temp = input_data["Air temperature [K]"].iloc[0]
    process_temp = input_data["Process temperature [K]"].iloc[0]

    if process_temp <= air_temp:
        errors.append(
            "Process Temperature must be higher than "
            "Air Temperature."
        )

    return errors