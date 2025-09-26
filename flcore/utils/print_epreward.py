def format_episode_info(ep, ep_reward, ep_info):
    # 格式化基础信息
    lines = [f"Episode: {ep + 1}"]

    for ep in ep_reward:
        lines.append(f"Reward: {ep:.3f},")
    lines.append(f"sum: {ep_reward.sum():.3f},")
    lines.append("  info: {")
    # 遍历字典，格式化每个键值对
    for key, value in ep_info.items():
        # 处理numpy类型，转为普通数值
        if hasattr(value, 'item'):
            value = value.item()

        # 根据值的类型进行格式化
        if isinstance(value, float):
            # 浮点数保留3-6位小数（根据数值大小自动调整）
            if abs(value) < 1e-3:
                formatted_value = f"{value:.6f}"
            else:
                formatted_value = f"{value:.3f}"
        else:
            formatted_value = str(value)

        lines.append(f"    '{key}': {formatted_value},")

    lines.append("  }")
    return '\n'.join(lines)