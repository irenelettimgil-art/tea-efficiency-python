## Project Overview

This project is a beginner-friendly toolkit that demonstrates how to analyze tea production efficiency using Python.
It applies a Cobb-Douglas production function to estimate the relationship between inputs (land, labor, fertilizer) and output.

## Requirements

- Python 3.x

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/irenelettimgil-art/tea-efficiency-python.git
```

2. Navigate into the folder:

```bash
cd tea-efficiency-python
```

3. Run the script:

```bash
python analysis.py
```

## Expected Output

- Regression results
- Efficiency scores for each tea farm

## AI Usage

This project was developed using generative AI prompts to:
- Translate R models into Python
- Generate dataset
- Debug code

## Prompt 1

Prompt Used:

> "Convert an R stochastic frontier model into Python using a simple approach."

AI Response Summary:
The AI suggested simplifying the stochastic frontier model into a Cobb-Douglas production function and using Python libraries such as `statsmodels` for estimation.

> "The AI recommended using an Ordinary Least Squares (OLS) regression model as an approximation of the production function."

It simplified a complex concept into a beginner-friendly implementation and made it possible to complete the project.

## Prompt 2

> "Explain how to calculate efficiency from regression residuals in Python"

Response Extract:

> "Efficiency can be approximated using the exponential of the residuals from the regression model."

It provided a practical way to estimate efficiency without using advanced stochastic frontier techniques.

## Prompt 3

> "Generate a realistic dataset for tea production, including land, labor, fertilizer, and output."

The AI generated a structured dataset representing tea farm inputs and outputs, suitable for analysis.

Response Extract:

> "The dataset includes variables such as land size, labor input, fertilizer use, and total output."

It saved time and allowed me to focus on analysis rather than data collection.

## Prompt 4

> "Help me debug a Python error when running statsmodels regression."

Response Extract:

> "Ensure the dataset is in the correct directory and all required libraries are installed using `pip`."

## Prompt 5

> "In the README.md, kindly format it well, without changing the content of the file, just formatting, e.g. code areas, like `pip install`, etc."

Response Extract:

> Done. Here's what was formatted (content unchanged):
> - `pip install`, `git clone`, `cd`, and `python analysis.py` are now in fenced `bash` code blocks
> - AI prompts are styled as blockquotes
> - Inline code formatting applied to `statsmodels` and `pip`
> - The files section is now a table
> - Fixed the numbering in "How to Run" (was 1, 3, 4)
> - Added consistent spacing between sections

I learned how to troubleshoot common Python errors and manage dependencies.
It improved my debugging skills and reduced the time spent resolving errors.

## Files Included

| File | Description |
|------|-------------|
| `tea_data.csv` | Dataset |
| `analysis.py` | Main code |
| `requirements.txt` | Dependencies |

## Author

Irene
