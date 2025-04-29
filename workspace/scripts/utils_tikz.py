import tempfile
import os
import subprocess
from PIL import Image


def render_tikz(tikz_code, output_path=None, scale=1.0, standalone=True):
    """Render TikZ code and save the resulting image.
    
    Args:
        tikz_code (str): The TikZ code to render
        output_path (str, optional): Path to save the output image. If None, a temporary file is used.
        scale (float): Scaling factor for the image
        standalone (bool): Whether to wrap the code in a standalone document
        
    Returns:
        str: Path to the rendered image
    """
    
    # Create a temporary directory for the LaTeX files
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = os.path.join(tmpdir, 'tikz_figure.tex')
        
        # Prepare the LaTeX document
        if standalone:
            full_code = (
                '\\documentclass[tikz]{standalone}\n'
                '\\usepackage{tikz}\n'
                '\\usepackage{amsmath,amssymb}\n'
                '\\usetikzlibrary{arrows,shapes,positioning,calc,decorations.pathreplacing,decorations.markings}\n'
                '\\begin{document}\n'
                f'{tikz_code}\n'
                '\\end{document}'
            )
        else:
            full_code = tikz_code
            
        # Write the LaTeX code to a file
        with open(tex_file, 'w') as f:
            f.write(full_code)
            
        # Compile the LaTeX file
        try:
            subprocess.run(['pdflatex', '-interaction=nonstopmode', '-output-directory', tmpdir, tex_file], 
                          check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Convert PDF to PNG
            pdf_file = os.path.join(tmpdir, 'tikz_figure.pdf')
            
            if output_path is None:
                # Create a temporary file for the output
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    png_file = tmp_file.name
            else:
                png_file = output_path
                
            dpi = 300 * scale
            
            subprocess.run(['convert', '-density', str(dpi), pdf_file, png_file],
                          check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return png_file
            
        except subprocess.CalledProcessError as e:
            print(f'Error rendering TikZ: {e}')
            print(f'Command output: {e.stdout.decode() if e.stdout else ""}')
            print(f'Command error: {e.stderr.decode() if e.stderr else ""}')
            return None


class TikZRenderer:
    def __init__(self, working_dir: str = ""):
        self.working_dir = working_dir
        self.file_idx = 0
        self.files = []
        os.makedirs(self.working_dir, exist_ok=True)
    
    def render(self, tikz_code: str):
        output_path = os.path.join(self.working_dir, f'tikz_figure_{self.file_idx}.png')
        self.file_idx += 1  
        file_path = render_tikz(tikz_code, output_path)
        if file_path is not None:
            self.files.append(file_path)
            return file_path
        else:
            return None
    
    def clear(self):
        for file in self.files:
            if os.path.exists(file):
                os.remove(file)
        self.files = []
        self.file_idx = 0


if __name__ == "__main__":
    tikz_code = """
    \\begin{tikzpicture}
    \\draw (0,0) -- (1,1);
    \\end{tikzpicture}
    """
    # render_tikz(tikz_code, "tikz_figure.png")
    renderer = TikZRenderer(working_dir='workspace/scripts/.temp')
    renderer.render(tikz_code)
    renderer.clear()
