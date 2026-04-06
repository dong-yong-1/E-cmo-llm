from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from dataset_schema import build_prompt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "dpo" / "small_dpo_pairs_v1.json"


def make_prompt(question: str, input_data: dict) -> str:
    return build_prompt({"instruction": f"用户问：{question}", "input": input_data})


def make_example(
    idx: int,
    category: str,
    subcategory: str,
    difficulty: str,
    question: str,
    input_data: dict,
    chosen: str,
    rejected: str,
    preference_type: str,
) -> dict:
    return {
        "id": f"dpo_{category}_{subcategory}_{idx:05d}",
        "category": category,
        "subcategory": subcategory,
        "difficulty": difficulty,
        "preference_type": preference_type,
        "prompt": make_prompt(question, input_data),
        "chosen": chosen,
        "rejected": rejected,
    }


def build_shipping_pairs(records: list[dict], start_idx: int) -> int:
    idx = start_idx

    transit_specs = [
        (
            "可以加急发货吗？",
            "您好，理解您着急收货的心情。您的订单已经在运输途中，目前无法再人工加急。建议您先查看订单页面的物流轨迹，如有异常我们也会继续协助跟进。",
            "您好，我们已经为您联系仓库加急处理，并会在24小时内反馈具体发货时间，请您放心。",
        ),
        (
            "为什么还没发货？",
            "您好，您的订单目前显示为运输中，说明商品已经从仓库发出，正在配送途中。您可以在订单页面查看最新物流轨迹，如有异常我们也会继续帮您跟进。",
            "您好，您的订单虽然在运输途中，但我们会继续帮您催仓并尽快安排发货，请您放心。",
        ),
        (
            "订单多久发货？",
            "您好，您的订单已经从仓库发出，目前正在运输途中。物流信息会持续更新，您可以在订单页面查看实时轨迹，我们也会继续关注配送进度。",
            "您好，商品已经发出后无法人工再加急发货，您可以持续关注物流轨迹，预计24小时内到达。",
        ),
    ]

    pending_specs = [
        (
            "可以加急发货吗？",
            "您好，理解您想尽快收到商品的心情。当前订单还是待发货状态，我们可以帮您催仓，但暂时无法保证具体发出时间，您可以留意订单页面的后续更新。",
            "您好，我们已经为您备注加急并联系仓库优先处理，预计今天一定会发出。",
        ),
        (
            "为什么还没发货？",
            "您好，当前订单还在待发货状态，我们可以帮您催仓跟进，但暂时无法保证具体发出时间。发货后物流信息会及时更新，辛苦您留意一下。",
            "您好，我们已经为您加急处理并重新打包，最晚24小时内一定发货。",
        ),
        (
            "订单多久发货？",
            "您好，您的订单目前处于待发货状态。通常发货后物流信息会更新，您也可以持续关注订单页面；我们这边可以帮您催仓，但暂时无法保证具体时效。",
            "您好，您的订单会在24小时内发出，如果没有发出我们会电话联系您说明情况。",
        ),
    ]

    for question, chosen, rejected in transit_specs:
        for product_category, emotion in product(["食品", "护肤品", "母婴用品"], ["neutral", "anxious"]):
            input_data = {
                "product_category": product_category,
                "order_status": "运输中",
                "is_opened": False,
                "has_quality_issue": False,
                "user_emotion": emotion,
                "platform_rule": "订单运输中无法人工加急，只能持续关注物流轨迹，不得承诺已开启紧急处理。",
            }
            records.append(
                make_example(
                    idx, "物流", "催发货", "medium", question, input_data, chosen, rejected, "safety_vs_overpromise"
                )
            )
            idx += 1

    for question, chosen, rejected in pending_specs:
        for product_category, emotion in product(["服饰", "数码配件", "家居用品"], ["neutral", "angry"]):
            input_data = {
                "product_category": product_category,
                "order_status": "待发货",
                "is_opened": False,
                "has_quality_issue": False,
                "user_emotion": emotion,
                "platform_rule": "订单待发货时可帮助催仓，但无法保证立即发出。",
            }
            records.append(
                make_example(
                    idx, "物流", "催发货", "medium", question, input_data, chosen, rejected, "natural_safe_vs_hard_promise"
                )
            )
            idx += 1

    return idx


def build_address_pairs(records: list[dict], start_idx: int) -> int:
    idx = start_idx
    shipped_specs = [
        (
            "发货后还能改地址吗？",
            "您好，订单发货后通常无法直接在后台修改地址。建议您先联系快递员或快递公司尝试协商转寄，具体结果需以物流方反馈为准。",
            "您好，订单发货后地址信息在订单页面右侧可以编辑。如果页面编辑不成功，您把收件信息发给我，我帮您进一步沟通处理。",
        ),
        (
            "能修改收货地址吗？",
            "您好，如果订单已经在运输途中，通常无法直接修改地址。建议您优先联系快递员或快递公司尝试协商转寄，如后续还有问题我们可以继续协助您了解处理路径。",
            "您好，当前订单已经发货，地址信息还可以在页面上编辑。如果改不了，我们也可以帮您恢复编辑入口。",
        ),
    ]
    pending_specs = [
        (
            "能修改收货地址吗？",
            "您好，当前订单还未发货，您可以先尝试在订单页面修改收货地址。如果页面暂时无法操作，我们也可以帮您进一步核实，但最终仍需以仓库处理结果为准。",
            "您好，当前订单还未发货，您直接点击页面下方的编辑按钮即可，改完后一定会立即生效。",
        ),
    ]

    for question, chosen, rejected in shipped_specs:
        for product_category, emotion in product(["护肤品", "食品", "家居用品"], ["neutral", "anxious"]):
            input_data = {
                "product_category": product_category,
                "order_status": "运输中",
                "is_opened": False,
                "has_quality_issue": False,
                "user_emotion": emotion,
                "platform_rule": "订单发货后通常无法直接修改地址，可建议联系派件员或快递协商，不得承诺已联系仓库改址。",
            }
            records.append(
                make_example(
                    idx, "物流", "地址修改", "medium", question, input_data, chosen, rejected, "safe_process_vs_false_capability"
                )
            )
            idx += 1

    for question, chosen, rejected in pending_specs:
        for product_category, emotion in product(["服饰", "母婴用品"], ["neutral", "anxious"]):
            input_data = {
                "product_category": product_category,
                "order_status": "待发货",
                "is_opened": False,
                "has_quality_issue": False,
                "user_emotion": emotion,
                "platform_rule": "订单待发货时可尝试修改地址，但修改结果需以仓库拦截或系统处理结果为准。",
            }
            records.append(
                make_example(
                    idx, "物流", "地址修改", "easy", question, input_data, chosen, rejected, "grounded_path_vs_false_ui"
                )
            )
            idx += 1

    return idx


def build_exchange_pairs(records: list[dict], start_idx: int) -> int:
    idx = start_idx
    specs = [
        (
            "可以换货吗？",
            "您好，如果商品存在质量问题，是可以申请换货的。请您先在订单页面提交售后申请并上传相关凭证，审核通过后我们会尽快为您安排后续处理。",
            "您好，如果商品存在破损或者损坏，您可以先暂时接受赔偿或优先换新。如果您不满意，我们会在24小时内为您直接安排换货。",
        ),
        (
            "商品坏了可以补发吗？",
            "您好，如果商品存在破损或质量问题，可以申请补发或换货。请您先在订单页面提交售后申请并上传相关凭证，审核通过后我们会尽快为您安排处理。",
            "您好，这边已经为您直接提交补发申请了，审批时间即时，后台会立刻开始换新流程。",
        ),
    ]
    for question, chosen, rejected in specs:
        for product_category, emotion in product(["母婴用品", "数码配件", "家居用品"], ["neutral", "anxious"]):
            input_data = {
                "product_category": product_category,
                "order_status": "已签收",
                "is_opened": True,
                "has_quality_issue": True,
                "user_emotion": emotion,
                "platform_rule": "换货或补发通常需要用户先在订单页面提交售后申请，审核通过后再安排处理，不得承诺固定联系时限。",
            }
            records.append(
                make_example(
                    idx, "售后", "换货补发", "easy", question, input_data, chosen, rejected, "proper_flow_vs_overpromise"
                )
            )
            idx += 1
    return idx


def build_refund_pairs(records: list[dict], start_idx: int) -> int:
    idx = start_idx
    specs = [
        (
            "不满意可以退款吗？",
            {
                "order_status": "已签收",
                "is_opened": True,
                "has_quality_issue": False,
                "platform_rule": "商品已拆封且非质量问题时，通常不支持7天无理由退货。",
            },
            "您好，如果商品已经拆封且没有质量问题，通常不支持7天无理由退货。若您担心商品存在异常，也可以把订单号和具体情况发给我，我帮您进一步核实。",
            "您好，根据平台规则，商品在收到后未拆封且没有质量问题的情况下，通常不支持7天无理由退货。请您放心，我会继续为您核实。",
        ),
        (
            "这个商品可以退吗？",
            {
                "order_status": "已签收",
                "is_opened": False,
                "has_quality_issue": False,
                "platform_rule": "商品未拆封且不影响二次销售，支持7天无理由退货。",
            },
            "您好，如果商品未拆封且不影响二次销售，一般支持7天无理由退货。您可以在订单页面提交退货申请，我们会按流程为您处理。",
            "您好，请您放心，我们已经为您提交了退货申请，24小时内会有人联系您处理。",
        ),
    ]
    for question, status_fields, chosen, rejected in specs:
        for product_category, emotion in product(["护肤品", "服饰", "母婴用品"], ["neutral", "anxious"]):
            input_data = {
                "product_category": product_category,
                "user_emotion": emotion,
                **status_fields,
            }
            records.append(
                make_example(
                    idx, "售后", "退货退款", "medium", question, input_data, chosen, rejected, "rule_boundary_vs_inverse_or_overpromise"
                )
            )
            idx += 1
    return idx


def build_product_pairs(records: list[dict], start_idx: int) -> int:
    idx = start_idx
    specs = [
        (
            "这个商品适合什么人群？",
            "您好，这款商品的具体适用人群建议您优先参考商品详情页中的商品说明或参数介绍。如果页面信息还不够清楚，您可以把商品链接发给我，我帮您进一步核实。",
            "您好，这款商品特别适合喜欢拍照和视频录制的消费者，具体效果也会比较明显。",
        ),
        (
            "库存还有吗？",
            "您好，商品库存是实时变动的，建议您以当前商品详情页显示的信息为准。如果您需要进一步确认，可以把商品名称或链接发给我，我帮您核实。",
            "您好，当前页面显示已经无货，建议您稍后再来看看。",
        ),
    ]
    for question, chosen, rejected in specs:
        for product_category, emotion in product(["数码配件", "护肤品", "家居用品"], ["neutral", "anxious"]):
            input_data = {
                "product_category": product_category,
                "order_status": "待发货",
                "is_opened": False,
                "has_quality_issue": False,
                "user_emotion": emotion,
                "platform_rule": "当上下文未提供明确商品属性时，应以商品详情页为准或说明可进一步核实，不得编造成分、材质、库存或适用人群。",
            }
            records.append(
                make_example(
                    idx, "售前", "商品咨询", "medium", question, input_data, chosen, rejected, "grounded_answer_vs_hallucination"
                )
            )
            idx += 1
    return idx


def build_complaint_pairs(records: list[dict], start_idx: int) -> int:
    idx = start_idx
    specs = [
        (
            "客服一直不回复",
            "非常抱歉给您带来了不好的体验。为了尽快帮您核实并跟进处理，麻烦您提供一下订单号和具体问题描述，我会先为您记录并继续跟进。",
            "非常抱歉给您带来了不好的体验，我将立刻为您转接至高级客服专员，我们会在24小时内通过消息或电话联系您。",
        ),
        (
            "我要投诉",
            "非常抱歉给您带来了不好的体验。为了尽快帮您核实并跟进处理，麻烦您提供一下订单号和具体问题描述，我会先为您记录并继续跟进。",
            "非常理解您焦急的心情，我这边已经为您联系负责人并安排专员跟进，稍后会通过电话联系您。",
        ),
        (
            "质量太差了怎么办？",
            "非常抱歉给您带来了不好的体验。商品目前还在运输途中，建议您收到后先检查。如果确认存在质量问题，您可以第一时间联系我们，我们会协助您处理退换货。",
            "非常抱歉给您带来了不好的体验。您可以直接申请退货退款，我们会尽快为您安排专员上门检测并处理。",
        ),
    ]
    for question, chosen, rejected in specs:
        for order_status, product_category, emotion in product(
            ["待发货", "运输中"], ["食品", "家居用品", "母婴用品"], ["anxious", "angry"]
        ):
            input_data = {
                "product_category": product_category,
                "order_status": order_status,
                "is_opened": order_status == "已签收",
                "has_quality_issue": False,
                "user_emotion": emotion,
                "platform_rule": "投诉场景需先安抚情绪，再给出明确处理路径；不得承诺具体反馈时限、电话短信通知或已安排专员。",
            }
            records.append(
                make_example(
                    idx, "投诉", "投诉安抚", "hard", question, input_data, chosen, rejected, "empathetic_grounded_vs_overpromise"
                )
            )
            idx += 1
    return idx


def main() -> None:
    records: list[dict] = []
    idx = 1
    idx = build_shipping_pairs(records, idx)
    idx = build_address_pairs(records, idx)
    idx = build_exchange_pairs(records, idx)
    idx = build_refund_pairs(records, idx)
    idx = build_product_pairs(records, idx)
    idx = build_complaint_pairs(records, idx)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(records)} pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
