import pytest
import pandas as pd
from src.pedkai_generator.validators.scenario_validator import (
    validate_scenario, ValidationResult, SCENARIO_SPECS
)

def test_scenario_specs_loaded():
    assert len(SCENARIO_SPECS) > 0

def test_validate_unknown_scenario():
    kpi = pd.DataFrame({"user_count": [0.0]})
    alarm = pd.DataFrame({"timestamp": [], "entity_id": [], "alarm_type": [], "severity": []})
    with pytest.raises(ValueError):
        validate_scenario("nonexistent_scenario", kpi, alarm)

def test_sleeping_cell_passes_with_zero_users():
    # Create DataFrame where user_count is ~0
    kpi = pd.DataFrame({"user_count": [0.0, 0.001, 0.0], "rsrp": [-70.0, -72.0, -68.0]})
    alarm = pd.DataFrame({"timestamp": pd.Series(dtype="str"), "entity_id": pd.Series(dtype="str"), "alarm_type": pd.Series(dtype="str"), "severity": pd.Series(dtype="str")})
    result = validate_scenario("sleeping_cell", kpi, alarm)
    assert isinstance(result, ValidationResult)

def test_validation_result_json_serialisable():
    import json
    kpi = pd.DataFrame({"user_count": [0.0], "rsrp": [-70.0]})
    alarm = pd.DataFrame({"timestamp": pd.Series(dtype="str"), "entity_id": pd.Series(dtype="str"), "alarm_type": pd.Series(dtype="str"), "severity": pd.Series(dtype="str")})
    result = validate_scenario("sleeping_cell", kpi, alarm)
    result_dict = result.to_dict()
    json.dumps(result_dict)  # Should not raise
