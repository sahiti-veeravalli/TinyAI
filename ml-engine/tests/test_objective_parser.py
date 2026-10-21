import pytest
from tinyai.optimization.objective_parser import ConversationalObjectiveParser


def test_parse_user_objective_specification():
    # Prompt explicitly specified in the user requirements
    prompt = (
        "Optimize this model for CPU inference. Keep macro F1 above 0.90, "
        "keep the model under 20 MB, and minimize median inference latency."
    )
    result = ConversationalObjectiveParser.parse(prompt)

    assert result.parser_type == "deterministic_rule_based"
    assert result.target_hardware in ["generic_cpu", "intel_nuc"]
    assert result.min_quality_score == 0.90
    assert result.quality_metric_name == "macro_f1"
    # 20 MB converted to KB: 20 * 1024 = 20480.0
    assert result.max_disk_size_kb == 20480.0
    assert result.priority == "lowest_latency"
    assert result.is_contradictory is False
    assert len(result.detected_entities) >= 3


def test_parse_rpi_and_microseconds():
    prompt = "Deploy on Raspberry Pi 4. Latency under 500us, footprint under 250 KB, ROC-AUC > 88%."
    result = ConversationalObjectiveParser.parse(prompt)

    assert result.target_hardware == "rpi4"
    assert result.max_latency_ms == 0.5  # 500us -> 0.5ms
    assert result.max_disk_size_kb == 250.0
    assert result.min_quality_score == 0.88
    assert result.quality_metric_name == "roc_auc"
    assert result.is_contradictory is False


def test_parse_cortex_m_smallest_size():
    prompt = "Optimize for ARM Cortex-M microcontroller. Shrink as much as possible, smallest size."
    result = ConversationalObjectiveParser.parse(prompt)

    assert result.target_hardware == "cortex_m"
    assert result.priority == "smallest_size"
    assert result.max_disk_size_kb <= 64.0
    assert result.is_contradictory is False


def test_detect_contradictory_constraints():
    prompt = "Latency under 0ms, quality above 1.5, size under -10KB."
    result = ConversationalObjectiveParser.parse(prompt)

    assert result.is_contradictory is True
    assert len(result.validation_warnings) >= 2


def test_explicit_strategy_filtering():
    prompt = "Only apply quantization and pruning on this LightGBM model, keep size < 100KB."
    result = ConversationalObjectiveParser.parse(prompt)

    assert "quantization" in result.enabled_strategies
    assert "pruning" in result.enabled_strategies
    assert "distillation" not in result.enabled_strategies
    assert result.max_disk_size_kb == 100.0
