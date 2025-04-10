import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from uitils import read_json_file
import numpy as np
import os
from typing import List
import argparse



def generate_html(input_string: str, importances: List[float], output_file: str = "visualize_token_importance/visualize_token_importance.html"):
    """
    Generate an HTML file to visualize the importance of characters using color gradients.

    Parameters:
        input_string (str): The original string.
        importances (List[float]): The importance of each character (normalized values).
        output_file (str): The name of the output HTML file.
    """
    # Skip spaces to calculate the effective character length
    non_space_chars = [char for char in input_string if char != ' ']
    effective_length = len(non_space_chars)

    # Scale the importance list to match the effective character length
    scaled_importances = np.interp(range(effective_length), np.linspace(0, effective_length - 1, len(importances)), importances)

    # Normalize the importance values
    normalized_importances = (np.array(scaled_importances) - np.min(scaled_importances)) / (np.max(scaled_importances) - np.min(scaled_importances))

    # Define the color range: the lowest value color is #618eff, the highest value color is #ff4d4d
    # min_color = np.array([0x61 / 255, 0x8e / 255, 0xff / 255])  # Convert to RGB ratio
    min_color = np.array([1, 1, 1])  # Convert to RGB ratio

    max_color = np.array([0xff / 255, 0x00 / 255, 0x00 / 255])

    # Generate HTML content
    html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Attention Visualization</title>
            <style>
                body {
                    background-color: #f0f0f0;
                    margin: 20px;
                    font-family: Arial, sans-serif;
                }
                .text-container {
                    font-family: Calibri;
                    font-size: 16px;
                    line-height: 1.2;
                    word-wrap: break-word;
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin: 0 auto;
                    max-width: 800px;
                }
                .char {
                    padding: 0 0.5px;  
                    margin: 0;
                    display: inline-block;
                    line-height: 1;
                    height: 1em;
                    letter-spacing: -1px;  
                }
                .space {
                    padding: 0 2px;  
                    display: inline-block;
                }
            </style>
        </head>
        <body>
            <div class="text-container">
        """

    importance_index = 0
    for char in input_string:
        if char == ' ':
            # Use wider spacing for spaces
            html_content += '<span class="space"> </span>'
        else:
            # Interpolate the color based on importance
            importance = normalized_importances[importance_index]
            importance_index += 1
            color = min_color * (1 - importance) + max_color * importance
            color_hex = mcolors.to_hex(color)  # Convert to hexadecimal color
            html_content += f'<span class="char" style="background-color: {color_hex};">{char}</span>'

    html_content += """
            </div>
        </body>
        </html>
        """

    # Write the HTML file
    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"HTML file generated: {os.path.abspath(output_file)}")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vicuna_7b_v1_5", help="The name of the LLM")
    parser.add_argument("--Jailbreak", type=str, default="GCG", help="The name of the Jailbreak")
    parser.add_argument("--BehaviorID", type=str, default="mexico_us_immigration_crime_link", help="The BehaviorID of the prompt")

    args = parser.parse_args()

    model_name = args.model_name
    Jailbreak = args.Jailbreak
    BehaviorID = args.BehaviorID

    # Load the defense results
    defense_results_file = "defense_results/defense_results_{}_{}.json".format(Jailbreak, model_name)
    defense_results = read_json_file(defense_results_file)
    if not defense_results:
        print(
            "{} file does not exist, run evaluation.py to create it first".format(defense_results_file))
        exit(-1)

    input_string = defense_results[BehaviorID]['prompt']
    importances = defense_results[BehaviorID]['token_importance']


    output_file = "visualize_token_importance/{}_{}_{}.html".format(Jailbreak, model_name, BehaviorID)

    generate_html(input_string, importances, output_file)