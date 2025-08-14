import torch
from pathlib import Path
import faiss
import gradio as gr
import traceback
import zstandard as zstd


# --- Gardio stuff start ---

def run_create_uvmp_script(pth_files, index_files, output_path):
    """Wrapper function to run the uvmp creation from the Gradio interface."""
    if not pth_files:
        return "Error: At least one .pth file is required."

    try:
        # Get the temporary file paths and sort them alphabetically for deterministic ID assignment
        pth_paths = sorted([f.name for f in pth_files])
        index_paths = [f.name for f in index_files] if index_files else []

        output_path = output_path if output_path else None

        return create_uvmp(pth_paths, index_paths, output_path)
    except Exception as e:
        return f"An unexpected error occurred: {e}\n{traceback.format_exc()}"


def uvmp_tab():
    """Defines the Gradio interface tab for the uvmp maker."""
    with gr.Column():
        gr.Markdown(
            """
            # UVMP Maker
            ### Unified Voice Model Package

            - Upload one or more `.pth` files and optionally their corresponding `.index` files. 
            - **For multi-model packages,** Speaker Names are derived from the .pth filenames (e.g., `speaker_a.pth` becomes speaker `speaker_a`).
            - If you provide `.index` files, their filenames must match their corresponding `.pth` file's name (e.g., `model_a.pth` and `model_a.index`).
            """
        )
        pth_input = gr.File(
            label="Upload PTH File(s)",
            file_types=[".pth"],
            file_count="multiple",
        )
        index_input = gr.File(
            label="Upload Index File(s) (Optional)",
            file_types=[".index"],
            file_count="multiple",
        )
        output_path_input = gr.Textbox(
            label="Output File Path (Optional)",
            info="If left blank, the .uvmp file will be saved in the main 'logs' directory.",
            placeholder="e.g., C:/logs/my_multi_model.uvmp",
            interactive=True,
        )
        uvmp_output_info = gr.Textbox(
            label="Output Information",
            info="The result of the operation will be displayed here.",
            value="",
            max_lines=8,
            interactive=False,
        )
        uvmp_create_button = gr.Button("Create UVMP File")
        
        uvmp_create_button.click(
            fn=run_create_uvmp_script,
            inputs=[pth_input, index_input, output_path_input],
            outputs=[uvmp_output_info],
        )

# --- Gardio stuff end ---

# --- uvmp creation start ---

def create_uvmp(pth_paths, index_paths=None, output_path=None):
    """
    Create a Zstandard-compressed .uvmp file from .pth and .index files.
    
    Speaker names are assigned based on the filename of the .pth files.
    """
    try:
        models_data = {}
        # Create a dictionary for quick lookup of index files by their base name
        index_paths_dict = {Path(p).stem: p for p in index_paths} if index_paths else {}

        # Iterate over the sorted .pth files to assign speaker names
        for pth_path in pth_paths:
            if not Path(pth_path).exists():
                return f"Error: PTH file not found: {pth_path}"

            pth_file_stem = Path(pth_path).stem
            # The speaker name is the filename without the extension
            speaker_name = pth_file_stem
            
            pth_data = torch.load(pth_path, map_location="cpu", weights_only=True)
            model_entry = {"model_state": pth_data}

            # Check if a corresponding index file was provided by matching filenames.
            if pth_file_stem in index_paths_dict:
                index_path = index_paths_dict[pth_file_stem]
                if not Path(index_path).exists():
                    return f"Error: Matching index file not found for {pth_file_stem}.pth at path: {index_path}"
                
                index = faiss.read_index(index_path)
                model_entry["index_data"] = faiss.serialize_index(index)
            
            models_data[speaker_name] = model_entry
        
        uvmp_data = {"models": models_data}

        # Determine the final output path for the .uvmp file.
        if output_path:
            final_output_path = Path(output_path)
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            logs_dir = project_root / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Use the base name of the first alphabetically sorted file for the output filename
            base_name = Path(pth_paths[0]).stem
            uvmp_filename = f"{base_name}_multi.uvmp" if len(pth_paths) > 1 else f"{base_name}.uvmp"
            final_output_path = logs_dir / uvmp_filename
            
        with zstd.open(final_output_path, 'wb') as f:
            torch.save(uvmp_data, f)
        
        return f"Successfully created uvmp file with {len(pth_paths)} model(s): {final_output_path}"
    except Exception as e:
        return f"An error occurred during uvmp creation: {e}\n{traceback.format_exc()}"
    
if __name__ == "__main__":
    with gr.Blocks() as demo:
        uvmp_tab()


# --- uvmp creation end ---