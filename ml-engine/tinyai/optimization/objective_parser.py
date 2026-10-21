import re
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class ParsedObjective(BaseModel):
    raw_prompt: str
    target_hardware: str = "generic_cpu"
    target_hardware_name: str = "Host General Purpose CPU"
    max_disk_size_kb: Optional[float] = 250.0
    max_latency_ms: Optional[float] = 1.0
    min_quality_score: Optional[float] = 0.85
    quality_metric_name: str = "macro_f1"
    max_memory_mb: Optional[float] = 100.0
    priority: str = "balanced"  # balanced, lowest_latency, smallest_size, highest_quality
    enabled_strategies: List[str] = Field(
        default_factory=lambda: ["quantization", "pruning", "distillation", "model_selection"]
    )
    detected_entities: List[Dict[str, Any]] = Field(default_factory=list)
    interpretation_summary: str = ""
    validation_warnings: List[str] = Field(default_factory=list)
    is_contradictory: bool = False
    parser_type: str = "deterministic_rule_based"
    confidence_score: float = 1.0


class ConversationalObjectiveParser:
    """
    Transparent, deterministic natural language objective parser.
    Translates user deployment objectives into validated optimization experiment configurations.
    Does not pretend to use an LLM: uses explicit semantic regex and token extraction rules.
    """

    HARDWARE_MAPPINGS = [
        (r"\b(?:rpi|raspberry\s*pi|pi\s*4|rpi4|arm64|cortex-a72)\b", "rpi4", "Raspberry Pi 4 (ARM Cortex-A72)"),
        (r"\b(?:cortex-m|mcu|microcontroller|stm32|nxp|bare\s*metal|embedded\s*sensor)\b", "cortex_m", "ARM Cortex-M (Microcontroller)"),
        (r"\b(?:jetson|jetson\s*nano|maxwell|gpu\s*edge|nvidia\s*edge)\b", "jetson_nano", "NVIDIA Jetson Nano (GPU Edge)"),
        (r"\b(?:intel\s*nuc|gateway|industrial\s*pc|x86_64|core\s*i5|atom|server)\b", "intel_nuc", "Intel Edge NUC / Industrial Gateway"),
        (r"\b(?:cpu|host\s*cpu|local\s*machine|desktop)\b", "generic_cpu", "Host General Purpose CPU"),
    ]

    @classmethod
    def parse(cls, prompt: str, current_constraints: Optional[Dict[str, Any]] = None) -> ParsedObjective:
        text = prompt.strip()
        detected_entities: List[Dict[str, Any]] = []
        warnings: List[str] = []
        is_contradictory = False

        # Defaults
        target_hw = "generic_cpu"
        target_hw_name = "Host General Purpose CPU"
        max_size_kb: Optional[float] = 250.0
        max_latency_ms: Optional[float] = 1.0
        min_quality: Optional[float] = 0.85
        quality_metric = "macro_f1"
        max_memory_mb: Optional[float] = 100.0
        priority = "balanced"
        strategies = ["quantization", "pruning", "distillation", "model_selection"]

        if current_constraints:
            if "max_disk_size_kb" in current_constraints:
                max_size_kb = current_constraints["max_disk_size_kb"]
            if "max_latency_ms" in current_constraints:
                max_latency_ms = current_constraints["max_latency_ms"]
            if "min_quality_score" in current_constraints:
                min_quality = current_constraints["min_quality_score"]
            if "priority" in current_constraints:
                priority = current_constraints["priority"]

        text_lower = text.lower()

        # 1. Target Hardware Extraction
        for pattern, hw_id, hw_name in cls.HARDWARE_MAPPINGS:
            if re.search(pattern, text_lower):
                target_hw = hw_id
                target_hw_name = hw_name
                detected_entities.append({
                    "entity": "target_hardware",
                    "value": hw_name,
                    "matched": pattern
                })
                # Set hardware-sensible defaults if user didn't specify strict numbers later
                if hw_id == "cortex_m":
                    max_size_kb = 64.0
                    max_latency_ms = 0.8
                    max_memory_mb = 1.0
                elif hw_id == "rpi4":
                    max_size_kb = 500.0
                    max_latency_ms = 3.0
                    max_memory_mb = 150.0
                elif hw_id == "jetson_nano":
                    max_size_kb = 2000.0
                    max_latency_ms = 1.5
                    max_memory_mb = 500.0
                elif hw_id == "intel_nuc":
                    max_size_kb = 5000.0
                    max_latency_ms = 0.5
                    max_memory_mb = 1024.0
                break

        # 2. Latency Budget Extraction
        # Look for e.g. "latency < 0.5ms", "latency under 20 ms", "keep under 1.2ms", "sub-0.30 ms", "0.2ms latency", "500us"
        latency_match = re.search(
            r"(?:latency|time|delay|sla|response)[^\d\w-]*(?:under|below|less\s+than|<=|<|max|at\s+most|sub-)?\s*(-?\d+(?:\.\d+)?)\s*(ms|s|sec|seconds|us|µs|microseconds)?",
            text_lower
        )
        if not latency_match:
            # Alternate pattern: e.g. "under 1.5ms"
            latency_match = re.search(r"\b(?:under|below|<=|<|max\s+of)?\s*(-?\d+(?:\.\d+)?)\s*(ms|µs|us)\b", text_lower)

        if latency_match:
            raw_val = float(latency_match.group(1))
            unit = latency_match.group(2) if latency_match.lastindex and latency_match.lastindex >= 2 else "ms"
            unit = unit.lower() if unit else "ms"

            if unit in ["s", "sec", "seconds"]:
                val_ms = raw_val * 1000.0
            elif unit in ["us", "µs", "microseconds"]:
                val_ms = raw_val / 1000.0
            else:
                val_ms = raw_val

            max_latency_ms = round(val_ms, 3)
            detected_entities.append({
                "entity": "max_latency_ms",
                "value": f"{max_latency_ms} ms",
                "extracted_raw": latency_match.group(0)
            })

        # 3. Model Size / Disk Budget Extraction
        # Look for e.g. "under 20 MB", "under 250 KB", "size < 500kb", "footprint below 64KB"
        size_match = re.search(
            r"(?:size|disk|footprint|storage|model\s+under|keep\s+the\s+model\s+under|under|below|<=|<)\s*(-?\d+(?:\.\d+)?)\s*(kb|mb|gb|bytes|b)\b",
            text_lower
        )
        if size_match:
            raw_size = float(size_match.group(1))
            size_unit = size_match.group(2).lower()
            if size_unit == "mb":
                val_kb = raw_size * 1024.0
            elif size_unit == "gb":
                val_kb = raw_size * 1024.0 * 1024.0
            elif size_unit in ["b", "bytes"]:
                val_kb = raw_size / 1024.0
            else:
                val_kb = raw_size

            max_size_kb = round(val_kb, 1)
            detected_entities.append({
                "entity": "max_disk_size_kb",
                "value": f"{max_size_kb} KB",
                "extracted_raw": size_match.group(0)
            })

        # 4. Quality Metric & Threshold Extraction
        # Look for e.g. "macro F1 above 0.90", "F1 > 0.85", "keep ROC-AUC above 0.92", "quality >= 85%"
        quality_match = re.search(
            r"(?:macro\s+f1|f1-score|f1|roc-auc|roc_auc|pr-auc|pr_auc|accuracy|quality|auc|score)[^\d\w-]*(?:above|over|greater\s+than|>=|>|min|at\s+least)?\s*(-?\d+(?:\.\d+)?)\s*(%)?",
            text_lower
        )
        if quality_match:
            raw_qual = float(quality_match.group(1))
            has_pct_symbol = quality_match.group(2) == "%"

            if has_pct_symbol:
                qual_val = round(raw_qual / 100.0, 4)
            elif raw_qual > 1.0 and raw_qual <= 100.0 and raw_qual >= 50.0:
                # E.g. "quality above 85" without % sign
                qual_val = round(raw_qual / 100.0, 4)
            else:
                qual_val = round(raw_qual, 4)

            min_quality = qual_val

            # Determine metric name
            matched_str = quality_match.group(0)
            if "pr" in matched_str:
                quality_metric = "pr_auc"
            elif "roc" in matched_str or "auc" in matched_str:
                quality_metric = "roc_auc"
            elif "f1" in matched_str:
                quality_metric = "macro_f1"
            elif "accuracy" in matched_str:
                quality_metric = "accuracy"

            detected_entities.append({
                "entity": "min_quality_score",
                "value": f"{min_quality} ({quality_metric})",
                "extracted_raw": matched_str
            })

        # 5. Priority / Objective Preference
        if re.search(r"\b(?:minimize\s+(?:median\s+)?(?:inference\s+)?latency|lowest\s+latency|fastest|sub-millisecond|ultra-low\s+latency|speed\s+first)\b", text_lower):
            priority = "lowest_latency"
            detected_entities.append({"entity": "priority", "value": "lowest_latency"})
        elif re.search(r"\b(?:minimize\s+(?:model\s+)?size|smallest\s+size|minimal\s+footprint|lowest\s+footprint|compact\s+first|maximum\s+compression)\b", text_lower):
            priority = "smallest_size"
            detected_entities.append({"entity": "priority", "value": "smallest_size"})
        elif re.search(r"\b(?:highest\s+quality|preserve\s+accuracy|maximum\s+f1|quality\s+first|no\s+quality\s+loss)\b", text_lower):
            priority = "highest_quality"
            detected_entities.append({"entity": "priority", "value": "highest_quality"})

        # 6. Specific Optimization Strategies
        explicit_strategies = []
        if re.search(r"\b(?:quantiz(?:ation|e)|fp16|float16|int8|precision)\b", text_lower):
            explicit_strategies.append("quantization")
        if re.search(r"\b(?:prun(?:ing|e)|sparsity|truncate|tree\s+pruning)\b", text_lower):
            explicit_strategies.append("pruning")
        if re.search(r"\b(?:distill(?:ation|ed|)?|student|teacher)\b", text_lower):
            explicit_strategies.append("distillation")
        if re.search(r"\b(?:shallow\s+tree|linear|model\s+selection|alternative\s+model)\b", text_lower):
            explicit_strategies.append("model_selection")

        if explicit_strategies:
            strategies = explicit_strategies
            detected_entities.append({
                "entity": "enabled_strategies",
                "value": ", ".join(explicit_strategies)
            })

        # 7. Constraint Validation & Contradiction Detection
        if max_latency_ms is not None:
            if max_latency_ms <= 0:
                warnings.append("Latency budget must be strictly greater than 0 ms.")
                is_contradictory = True
            elif max_latency_ms < 0.05:
                warnings.append(
                    f"Latency budget of {max_latency_ms}ms is near physical limits of Python runtime process scheduling on host CPU."
                )

        if max_size_kb is not None:
            if max_size_kb <= 0:
                warnings.append("Model size budget must be strictly positive.")
                is_contradictory = True
            elif max_size_kb < 5.0 and "model_selection" not in strategies:
                warnings.append(
                    f"A model size budget of {max_size_kb}KB is too restrictive for tree ensembles. Enable 'model_selection' for L1 linear models."
                )
            elif max_size_kb > 20480.0:  # > 20MB
                warnings.append(
                    f"Model size budget of {max_size_kb/1024:.1f}MB is generous for tabular edge models; standard edge limits are under 1MB."
                )

        if min_quality is not None:
            if min_quality < 0.0 or min_quality > 1.0:
                warnings.append(f"Quality score {min_quality} is outside the valid probability bounds [0.0, 1.0].")
                is_contradictory = True
            elif min_quality > 0.98:
                warnings.append(
                    f"Minimum quality requirement of {min_quality:.2f} is exceptionally high. Compression strategies may not be feasible without slight degradation."
                )

        # 8. Human-Readable Interpretation Summary
        summary_parts = [
            f"Deployment Target: {target_hw_name}",
            f"Latency SLA: <= {max_latency_ms:.3f} ms" if max_latency_ms else "Latency SLA: Unbounded",
            f"Size Footprint: <= {max_size_kb:.1f} KB" if max_size_kb else "Size Footprint: Unbounded",
            f"Quality Threshold: >= {min_quality:.4f} ({quality_metric})" if min_quality else "Quality: No minimum",
            f"Ranking Priority: {priority.replace('_', ' ').title()}",
            f"Active Strategies: {', '.join(strategies)}"
        ]
        interpretation_summary = " | ".join(summary_parts)

        return ParsedObjective(
            raw_prompt=prompt,
            target_hardware=target_hw,
            target_hardware_name=target_hw_name,
            max_disk_size_kb=max_size_kb,
            max_latency_ms=max_latency_ms,
            min_quality_score=min_quality,
            quality_metric_name=quality_metric,
            max_memory_mb=max_memory_mb,
            priority=priority,
            enabled_strategies=strategies,
            detected_entities=detected_entities,
            interpretation_summary=interpretation_summary,
            validation_warnings=warnings,
            is_contradictory=is_contradictory,
            parser_type="deterministic_rule_based",
            confidence_score=0.95 if detected_entities else 0.70
        )
