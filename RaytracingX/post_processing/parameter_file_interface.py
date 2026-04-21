import numpy as np

def remove_whitespace(string : str) -> str:
    return "".join(string.split())

def remove_brackets(string, brackets = "[]") -> str:
    brackets = remove_whitespace(brackets)
    if len(brackets) != 2:
        raise ValueError("invalid string literal for remove_brackets(): " + brackets)
    return string.split(brackets[0])[1].split(brackets[1])[0]

def read_parameter(string : str, true_list = ["true", "True", "yes", "Yes"], false_list = ["false", "False", "no", "No"], check_for_quotation_marks = True):
    if np.any(np.array([true_list[i] in string for i in range(len(true_list))])):
        return True
    if np.any(np.array([false_list[i] in string for i in range(len(true_list))])):
        return False

    for TYPE in [int, float]:
        try:
            return TYPE(string)
        except ValueError:
            continue
    
    if check_for_quotation_marks:
        if string.count("\"") == 2:
            return remove_brackets(string, brackets = "\"\"")
        if string.count("\'") == 2:
            return remove_brackets(string, brackets = "\'\'")
    
    raise ValueError("string \"" + string + "\" did not match any of the supported types: bool, int, float" + ", str" if check_for_quotation_marks else "")

def read_parameter_file(parfile : str, thorn_name : str) -> dict:
    par_dict = {}

    for line in open(parfile):
        if line[:len(thorn_name)] != thorn_name:
            continue

        comment_index = line.find("#")
        if comment_index != -1:
            line = line[:comment_index]

        line = line[len(thorn_name) + 2:]

        linesplit = line.split("=")
        par_name = remove_whitespace(linesplit[0])

        line = linesplit[1]

        if "[" in line:
            line = remove_brackets(line)
            linesplit = line.split(",")
            par_dict[par_name] = np.array([read_parameter(linesplit[i]) for i in range(len(linesplit))])
        else:
            par_dict[par_name] = read_parameter(line)
    return par_dict