import json

def extract_output_values_from_json_file(file):
    '''
    Reads a file containing JSON-formatted results and returns a list of output values.

    Parameters:
    - file (str): The file path to the JSON results file.

    Returns:
    list: A list containing the 'output' values extracted from each line of the JSON file.

    Example:
    results_list = extract_output_values_from_json_file('/path/to/results.json')
    '''
    results_list = []  
    with open(file, 'r') as infile:
        results = infile.read()
        results = results.strip().split('\n')
        for line in results:
            results_list.append(json.loads(line)['output'])
    return results_list