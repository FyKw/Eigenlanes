import csv

input_file = "test_layer_times.csv"
output_file = "test_layer_times_clean.csv"
# input_file = "test_log.csv"
# output_file = "test_log_clean.csv"

# Example: column 1 = names, column 2 = seconds, column 3 = percentages
# Use zero-based indexing
#percentage_columns = {3,4,5}   # set of column indices that are percentages
percentage_columns = {}

with open(input_file, "r", newline="") as infile, open(output_file, "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        new_row = []
        for idx, cell in enumerate(row):
            try:
                value = float(cell)
                if idx in percentage_columns:
                    # Store as number (12.34 instead of 0.1234)
                    # Sheets will see it as 12.34, then you can format as %
                    new_row.append(f"{value * 100:.1f}")
                else:
                    # Leave time or other floats untouched
                    new_row.append(f"{value:.3f}")  # unify float formatting
            except ValueError:
                # Not a number (string like a name)
                new_row.append(cell)
        writer.writerow(new_row)
