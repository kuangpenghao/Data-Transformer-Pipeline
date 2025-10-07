import subprocess

def list_tmux_sessions():
    """
    列出所有当前的 tmux 会话
    :return: 会话名称列表
    """
    result = subprocess.run(
        ["tmux", "ls"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        error_msg = result.stderr.decode().strip()
        print(f" 无法获取会话列表: {error_msg}")
        return []

    # 解析每一行，只保留冒号前的会话名
    lines = result.stdout.decode().strip().split('\n')
    sessions = [line.split(':')[0] for line in lines if line]
    return sessions

def print_tmux_sessions():
    sessions=list_tmux_sessions()
    for session in sessions:
        print(session)

def close_tmux_session(session_name):
    """
    关闭指定名称的 tmux 会话（包括所有窗口）
    :param session_name: 会话名称
    """
    print(f"正在尝试关闭会话 '{session_name}' ...")
    result = subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode == 0:
        print(f" 成功关闭会话 '{session_name}'")
    else:
        error_msg = result.stderr.decode().strip()
        print(f" 无法关闭会话 '{session_name}': {error_msg}")

def create_tmux_session(session_name):
    """
    创建指定名称的 tmux 会话（包括所有窗口）
    :param session_name: 会话名称
    """
    print(f"正在尝试打开会话 '{session_name}' ...")
    result = subprocess.run(
        ["tmux", "new", "-s", session_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode == 0:
        print(f" 成功打开会话 '{session_name}'")
    else:
        error_msg = result.stderr.decode().strip()
        print(f" 无法打开会话 '{session_name}': {error_msg}")

def attach_tmux_session(session_name):
    result = subprocess.run(
        ["tmux", "attach", "-t", session_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode == 0:
        print(f" 成功进入会话 '{session_name}'")
    else:
        error_msg = result.stderr.decode().strip()
        print(f" 无法进入会话 '{session_name}': {error_msg}")

def submit_command(session_name: str,command_line) -> bool:
    try:
        # 向tmux会话发送命令
        result = subprocess.run(
            ["tmux", "send-keys", "-t", session_name, command_line, "Enter"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        success = result.returncode == 0
        if success:
            print(f" 成功向会话 {session_name} 提交训练任务")
            return True
        else:
            print(f" 向会话 {session_name} 提交任务失败")
            return False
    except Exception as e:
        print(f" 提交任务时出错: {e}")
        return False

if __name__ == "__main__":

    '''
    sessions = list_tmux_sessions()
    
    print(sessions)
    '''

    ''''''
    sessions=[
        "5M",
        "21M",
        "276M",
        "2G",
        "11G"
    ]

    for i,session in enumerate(sessions):
        '''
        command_line = (
            f"conda activate pt && "
            f"slash activate kphtemp && "
            f"export CUDA_VISIBLE_DEVICES={i} && "
            f"wandb agent --count 1 kuangpenghao-shanghaitech-university/Probabilistic_Transformer_KPH/29v39rxq"
        )
        submit_command(session, command_line)
        '''
        '''
        create_tmux_session(session)
        '''
        
        attach_tmux_session(session)
        ''''''
# python -u "train_scripts/tmux_script.py"
# export CUDA_VISIBLE_DEVICES=2 && wandb agent --count 3 kuangpenghao-shanghaitech-university/Probabilistic_Transformer_KPH/j0o871p8