import os, sys
import pickle
from autogen.coding import CodeBlock
from autogen.coding.jupyter import JupyterCodeExecutor, LocalJupyterServer
import ast, re
import subprocess
import tempfile
import os

# add the tools directory to the path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# for each dialogue, we will have a new code executor
class CodeExecutor:
    def __init__(
        self, 
        working_dir: str = "",
        use_vision_tools: bool = False,
        ):
        self.working_dir = working_dir
        
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)
            
        # set up the server
        self.server = LocalJupyterServer()
            
        # set up the jupyter executor
        self.executor = JupyterCodeExecutor(self.server, output_dir=self.working_dir)
        
        # initialize the environment
        self.init_env(use_vision_tools)
        self.file_idx = 0
        
    def result_processor(self, result):
        # Change an IPythonCodeResult object to a string, and the list of files
        # If the execution failed, the string is the error message.
        # If the execution is successful, the string is the output of the code execution.
        # In the string, all embeded PIL images are replaced by their file paths, using html img tag.
        # The list of files are the paths of the images.
        
        # process error message
        def parse_error_message(error):
            # Find the index where the list starts, indicated by `['`
            list_start_index = error.find("['")
            
            # The first part before the list is the initial error message
            initial_error = error[:list_start_index].strip()
            
            # The second part is the list of strings, which starts from `['` and goes to the end of the string
            traceback_list_str = error[list_start_index:]
            
            # Use ast.literal_eval to safely evaluate the string representation of the list
            # This method is safer than eval and can handle Python literals properly
            try:
                traceback_list = ast.literal_eval(traceback_list_str)
            except SyntaxError as e:
                print("Error parsing the list: ", e)
                traceback_list = []
                
            # Remove ANSI escape sequences
            ansi_escape = re.compile(r'\x1b\[.*?m')
            traceback_list = [ansi_escape.sub('', line) for line in traceback_list]
            
            return initial_error + "\n\n" + "\n".join(traceback_list)
        
        
        exit_code = result.exit_code
        
        file_paths = result.output_files
        output_str = result.output
        output_lines = output_str.split("\n")
        
        if len(file_paths) > 0:
            output_lines = output_lines[:-2*len(file_paths)]
        
        # replace the file name with the file index
        for _, file_path in enumerate(file_paths):
            dir_name = os.path.dirname(file_path)
            new_file_path = f"{dir_name}/file_{self.file_idx}.png"
            os.rename(file_path, new_file_path)
            file_paths[_] = new_file_path
            self.file_idx += 1
            
        # if execution succeeded, replace PIL images with their file paths
        if exit_code == 0:
            new_str = ""
            image_idx = 0
            
            for line in output_lines:
                if line.startswith("<PIL."):
                    if image_idx < len(file_paths):
                        new_str += f"<img src='{file_paths[image_idx]}'>"
                        image_idx += 1
                else:
                    new_str += line
                new_str += "\n"
            
            # add the remaining images
            for file_idx, file in enumerate(file_paths):
                if file_idx >= image_idx:
                    new_str += f"<img src='{file}'>"
                    new_str += "\n"
                
            return exit_code, new_str, file_paths
        
        # if execution failed, parse the error message
        else:
            error_msg = parse_error_message(output_str)
            return exit_code, error_msg, file_paths
    
    def execute(self, code: str):
        # Add default figure size control and axis scaling to prevent oversized plots

        code_with_size_control = code
 
        self.executor._jupyter_kernel_client = self.executor._jupyter_client.get_kernel_client(self.executor._kernel_id)
        execution_result = self.executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python",
                        code=code_with_size_control),
            ]
        )
        ret = self.result_processor(execution_result)
        return ret
    
    def init_env(self, use_vision_tools):
        init_code = ("import sys\n"
                     "from PIL import Image\n"
                     "from IPython.display import display\n"
                     f"sys.path.append('{parent_dir}')\n"
                     "from utils_geometry import *\n"
        )

        init_resp = self.execute(init_code)
        print(init_resp[1])


    def cleanup(self):
        self.server.stop()


def run_python_code(code, working_dir=None):
    """
    在子进程中运行Python代码并返回状态码、标准输出和标准错误
    
    参数:
        code (str): 要执行的Python代码字符串
        working_dir (str, optional): 运行代码的工作目录。默认为None，表示使用当前目录
    
    返回:
        tuple: (状态码, 标准输出, 标准错误)
            - 状态码 (int): 执行代码的进程退出状态码
            - 标准输出 (str): 进程的标准输出内容
            - 标准错误 (str): 进程的标准错误内容
    """
    # 创建临时文件来存储代码
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        temp_filename = temp_file.name
        # 将代码写入临时文件
        temp_file.write(code.encode('utf-8'))
    
    try:
        # 在子进程中运行Python代码，指定工作目录
        process = subprocess.Popen(['python', temp_filename], 
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  cwd=working_dir)  # 设置工作目录
        
        # 等待进程完成并获取输出
        stdout, stderr = process.communicate()
        
        # 将字节转换为字符串
        stdout_str = stdout.decode('utf-8')
        stderr_str = stderr.decode('utf-8')
        
        return process.returncode, stdout_str, stderr_str
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


class SandboxCodeExecutor:
    def __init__(
        self, 
        working_dir: str = os.path.join(os.path.dirname(__file__), '.temp/sandbox_code_executor'),
        use_vision_tools: bool = False,
        ):
        self.working_dir = working_dir
        
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)
            
        # Initialize environment variables
        self.globals = {}
        self.locals = {}
        self.file_idx = 0
        
        # Initialize the environment
        self.init_env(use_vision_tools)
    
    def next_file_path(self):
        new_file_path = f"{self.working_dir}/file_{self.file_idx}.png"
        self.current_file_path = new_file_path
        self.file_idx += 1
        return new_file_path

        
    def result_processor(self, result):
        # Process result in the same way as the original CodeExecutor
        exit_code, output_str, file_paths = result
        
        if len(file_paths) > 0:
            output_lines = output_str.split("\n")
            output_lines = output_lines[:-2*len(file_paths)]
            output_str = "\n".join(output_lines)
    
        # If execution succeeded, replace PIL images with their file paths
        if exit_code == 0:
            new_str = ""
            image_idx = 0
            
            for line in output_str.split("\n"):
                if line.startswith("<PIL."):
                    if image_idx < len(file_paths):
                        new_str += f"<img src='{file_paths[image_idx]}'>"
                        image_idx += 1
                else:
                    new_str += line
                new_str += "\n"
            
            # Add the remaining images
            for file_idx, file in enumerate(file_paths):
                if file_idx >= image_idx:
                    new_str += f"<img src='{file}'>"
                    new_str += "\n"
                
            return exit_code, new_str, file_paths
        
        # If execution failed, return the error message
        else:
            return exit_code, output_str, file_paths
    
    def execute(self, code: str):
        import io
        import sys
        import traceback
        from contextlib import redirect_stdout, redirect_stderr
        
        # Create file paths list to track generated images
        file_paths = []
        # Execute the code in the sandbox
        exit_code = 0

        try:
            # Add figure size control similar to original
            assert 'plt.show()' in code, "Code must contain plt.show()"
            if 'plt.show()' in code:
                code = code.replace('plt.show()', f'plt.savefig("{self.next_file_path()}")')

            code_with_size_control = self.init_code + code
            # print(code_with_size_control)

            exit_code, stdout, stderr = run_python_code(code_with_size_control, working_dir=None)

            assert exit_code == 0, f"Error: {stderr}"
            file_paths.append(self.current_file_path)
            return self.result_processor((exit_code, stdout, file_paths))


        except Exception as err:
            output = f"Error: {str(err)}"
            # Process the output
            return self.result_processor((1, output, []))
    

    def init_env(self, use_vision_tools):
        self.init_code = ("import sys\n"
                     "import matplotlib.pyplot as plt\n"
                     "from PIL import Image\n"
                     "from IPython.display import display\n"
                     f"sys.path.append('{parent_dir}')\n"
                     "from utils_geometry import *\n"
        )

    def cleanup(self):
        # Clean up resources if needed
        self.globals.clear()
        self.locals.clear()


if __name__ == "__main__":
    executor = CodeExecutor('workspace/scripts/.temp/code_executor')
    sandbox_executor = SandboxCodeExecutor('workspace/scripts/.temp/sandbox_code_executor')

    code = "\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# --- Helper functions ----------------------------------------------------\ndef draw_polygon(ax, points, **kwargs):\n    poly = plt.Polygon(points, closed=True, edgecolor='blue', fill=False, linewidth=2, **kwargs)\n    ax.add_patch(poly)\n\ndef draw_lines(ax, p1, p2, **kwargs):\n    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue', linewidth=2, **kwargs)\n\ndef draw_angle_marker(ax, vertex, p1, p2, label=None, color='gray', size_ratio=0.18, **kwargs):\n    # Draw an angle arc at 'vertex' between [p1, vertex, p2]\n    # Calculate vectors\n    v1 = np.array(p1) - np.array(vertex)\n    v2 = np.array(p2) - np.array(vertex)\n    ang1 = np.arctan2(v1[1], v1[0])\n    ang2 = np.arctan2(v2[1], v2[0])\n    # Ensure ang2 > ang1\n    if ang2 < ang1:\n        ang1, ang2 = ang2, ang1\n    angle = ang2 - ang1\n    arc_radius = size_ratio * max(np.linalg.norm(v1), np.linalg.norm(v2))\n    arc = patches.Arc(vertex, 2*arc_radius, 2*arc_radius,\n                      theta1=np.degrees(ang1), theta2=np.degrees(ang2),\n                      color=color, lw=2)\n    ax.add_patch(arc)\n    # Place the label\n    angle_middle = (ang1 + ang2) / 2\n    label_radius = arc_radius * 1.25\n    label_x = vertex[0] + label_radius * np.cos(angle_middle)\n    label_y = vertex[1] + label_radius * np.sin(angle_middle)\n    if label is not None:\n        ax.text(label_x, label_y, label, fontsize=12, ha='center', va='center', color='black')\n\n# --- Parallelogram Construction ------------------------------------------\nM = np.array([60, 0])\nN = np.array([260, 0])\nR = np.array([0, 150])\n# P = N + (R - M)\nP = N + (R - M)\n# intersection of diagonals: Q = (M + P)/2 = (N + R)/2\nQ = (M + P) / 2\n\npoints = {'M': M, 'N': N, 'P': P, 'R': R, 'Q': Q}\n\nfig, ax = plt.subplots(figsize=(6, 6))\n\n# Draw parallelogram\ndraw_polygon(ax, [M, N, P, R])\n\n# Draw diagonals\ndraw_lines(ax, M, P)\ndraw_lines(ax, N, R)\n\n# --- Angle Markers (see diagram) -----------------------------------------\ndraw_angle_marker(ax, R, M, P, label='38°', color='gray', size_ratio=0.18)\ndraw_angle_marker(ax, Q, R, P, label='3z - 3', color='gray', size_ratio=0.18)\ndraw_angle_marker(ax, Q, M, N, label='4w - 3', color='gray', size_ratio=0.16)\ndraw_angle_marker(ax, N, M, P, label='33°', color='gray', size_ratio=0.16)\ndraw_angle_marker(ax, Q, N, P, label='83°', color='gray', size_ratio=0.18)\n\n# --- Segment Labels ------------------------------------------------------\ndef midpoint(a, b, offset=(0,0)):\n    return ((a[0]+b[0])/2 + offset[0], (a[1]+b[1])/2 + offset[1])\n\nax.text(*midpoint(M,N, (0,-12)), '3x-4', fontsize=12, ha='center')\nax.text(*midpoint(N,P, (24,0)), '15.4', fontsize=12, ha='center')\nax.text(*midpoint(R,M, (-18,0)), '17.9', fontsize=12, ha='center')\nax.text(*midpoint(R,P, (0,14)), '20', fontsize=12, ha='center')\nax.text(*midpoint(N,P, (38,14)), '2y+5', fontsize=12, ha='center')\nax.text(*midpoint(Q,P, (16,2)), '11.1', fontsize=12, ha='center')\n\n# --- Point Labels --------------------------------------------------------\nax.text(M[0], M[1]-12, 'M', fontsize=12, ha='center')\nax.text(N[0], N[1]-12, 'N', fontsize=12, ha='center')\nax.text(P[0]+12, P[1]+6, 'P', fontsize=12, ha='center')\nax.text(Q[0]-14, Q[1]+0, 'Q', fontsize=12, ha='center')\nax.text(R[0]-10, R[1]+10, 'R', fontsize=12, ha='center')\n\n# --- Formatting ----------------------------------------------------------\nax.set_aspect('equal')\nax.set_xticks([])\nax.set_yticks([])\nfor spine in ax.spines.values():\n    spine.set_visible(False)\n\nplt.tight_layout()\nplt.show()\n"

    problematic_code = "\n# Import the necessary packages\nimport matplotlib.pyplot as plt\nfrom matplotlib.patches import Polygon\nimport matplotlib.patches as patches\n\n# Point Definition\npoints = {\n    'A': (105.0, 1.0),\n    'B': (198.0, 68.0),\n    'C': (107.0, 187.0),\n    'D': (1.0, 68.0)\n}\n\n# Ceate a figure and an axis\nfig, ax = plt.subplots()\n\n# Draw the polygon\npolygon = Polygon([points['A'], points['B'], points['C'], points['D']], closed=True, edgecolor='blue', fill=False)\nax.add_patch(polygon)\n\n# Draw angle markers using arcs\nax.add_patch(patches.Arc(points['A'], 20, 20, angle=0, theta1=180, theta2=300, color='pink'))\nax.add_patch(patches.Arc(points['D'], 20, 20, angle=0, theta1=270, theta2=355, color='pink'))\n\n# Annotate the points\nax.text(points['A'][0], points['A'][1] - 10, 'A', fontsize=12, ha='center')\nax.text(points['B'][0] + 5, points['B'][1], 'B', fontsize=12, ha='center')\nax.text(points['C'][0], points['C'][1] + 10, 'C', fontsize=12, ha='center')\nax.text(points['D'][0] - 10, points['D'][1], 'D', fontsize=12, ha='center')\n\n# Annotate the angles\nax.text(points['A'][0] + 20, points['A'][1] + 5, '120°', fontsize=12, ha='center', color='green')\nax.text(points['D'][0] + 10, points['D'][1] - 5, '85°', fontsize=12, ha='center', color='green')\n\n# Set figure format\nax.set_aspect('equal')\nax.set_xticks([])\nax.set_yticks([])\nax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False)\nax.spines['bottom'].set_visible(False)\nax.spines['left'].set_visible(False)\n\nplt.show()\n"
    # result = executor.execute(problematic_code)
    result = sandbox_executor.execute(problematic_code)
    print(result[0])
    print(result[1])
