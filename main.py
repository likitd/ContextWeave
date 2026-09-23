import asyncio
import time
from pathlib import Path
import config
from rag_pipeline import HybridRAGPipeline

async def main():
    pipeline = HybridRAGPipeline()
    await pipeline.initialize()

    instruction_path = Path("instructions/system_instruction.txt")
    output_format_path = Path("instructions/output_format.json")
    system_instruction = instruction_path.read_text(encoding="utf-8")
    output_format = output_format_path.read_text(encoding="utf-8")

    user_request = """Please generate the Android UI metadata JSON for a dialog box representing an 'Add Primary Skills' form. 
The layout should be structured from top to bottom exactly as follows:

1. **Header:** - A title "Add Primary Skills" aligned to the left.
   - A close icon ('X') aligned to the right.

2. **Skill Selection:** - A dropdown/select component. 
   - The main label is "Select Primary Skill" with secondary muted text next to it saying "(only 1)".
   - The selected value inside the dropdown is "UI / UX".

3. **Rating Component:** - A label "Rating".
   - Below the label, a row of exactly 10 unfilled/empty star icons.
   - Below the stars, a small text row showing "Rating:" on the left and a hyphen "-" on the right.

4. **Request Button:** - A full-width button with a light blue background and blue text that says "Request Add New Skill".

5. **Action Buttons:** - A horizontal row containing two buttons side-by-side.
   - The left button is "Cancel" with an outlined style (blue border, white background, blue text).
   - The right button is "Add" with a solid grey background and white text, representing a disabled state.

6. **Information Banner:** - A warning/info box at the very bottom with a light red/pink background.
   - It has a red circular info 'i' icon on the left.
   - The text inside has two parts:
     - Part 1 (Bold red text): "Rating Legend: 1-3: Beginner | 4-5: Intermediate | 6-7: Advanced | 8-10: Expert"
     - Part 2 (Italic red text below it): "Ratings 8-10 (Expert Level) require manager approval before they are applied. On rejection, the rating defaults back to 7."

Ensure the generated metadata strictly matches this layout hierarchy and includes all text exactly as provided."""

    query = f"""[INSTRUCTIONS]
{system_instruction}

[OUTPUT FORMAT / SCHEMA]
{output_format}

[USER REQUEST]
{user_request}"""

    answer = await pipeline.query(query)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    timestamp = int(time.time())
    strategy = getattr(config, "STRATEGY", "unknown")
    out_path = output_dir / f"{strategy}_{timestamp}.txt"
    out_path.write_text(answer, encoding="utf-8")
    await pipeline.get_pipeline_stats()
    await pipeline.shutdown()

def run():
    asyncio.run(main())

if __name__ == "__main__":
    run()
