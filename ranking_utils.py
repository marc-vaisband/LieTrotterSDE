"""
Various utilities for automatically generating LaTeX tables from the results of the experiments,
including reporting rank sums of the models on the different problems.
"""

def get_ranks(problem, methods, results_dict):
    problem_entries = {
        method: results_dict[(problem, method)] for method in methods
    }
    
    metrics = list(problem_entries[methods[0]].keys())
    
    ranks = {
        metric: {
            method: 0 for method in methods
        } for metric in metrics
    }
    
    for metric in metrics:
        sorted_methods = sorted(methods, key=lambda method: problem_entries[method][metric])
        for i, method in enumerate(sorted_methods):
            ranks[metric][method] = i+1
            
    return ranks

def sum_ranks(problem, methods, results_dict):
    ranks = get_ranks(problem, methods, results_dict)
    
    metrics = list(ranks.keys())
    
    summed_ranks = {
        method: 0 for method in methods
    }
    
    for metric in metrics:
        for method in methods:
            summed_ranks[method] += ranks[metric][method]
            
    return summed_ranks

def sum_all_ranks(methods, results_dict):
    summed_ranks = {
        method: 0 for method in methods
    }
    
    problems_in_dict = set([problem for (problem, method) in results_dict.keys()])
    
    for problem in problems_in_dict:
        summed_ranks_problem = sum_ranks(problem, methods, results_dict)
        for method in methods:
            summed_ranks[method] += summed_ranks_problem[method]
            
    return summed_ranks

def get_ranking(methods, results_dict):
    summed_ranks = sum_all_ranks(methods, results_dict)
    
    sorted_methods = sorted(methods, key=lambda method: summed_ranks[method])
    
    return sorted_methods

method_to_nice_method = {
    "euler": "Euler-Maruyama (EM)",
    "lt": "Lie-Trotter (LT)",
    "moment": "Moment (M) \cite{rackauckas2020universal}",
    "wasserstein": "Wasserstein (W)",
    "corr": "Auto-Correlation (C)",
    "euler_moment": "EM + M",
    "euler_wasserstein": "EM + W",
    "euler_corr": "EM + C",
    "lt_moment": "LT + M",
    "lt_wasserstein": "LT + W",
    "lt_corr": "LT + C",
    "euler_moment_corr": "EM + M + C",
    "euler_wasserstein_corr": "EM + W + C",
    "lt_moment_corr": "LT + M + C",
    "lt_wasserstein_corr": "LT + W + C",
    "sdegan": "SDE-GAN \cite{kidger2021neural}"
}

problem_to_nice_problem = {
    "ou": "OU",
    "cir": "CIR",
    "sin": "SIN1",
    "sit": "SIN2",
    "gfp": "GFP"
}

def write_latex_table(problems_listed, methods, results_dict, out_txt_file, cell="rank"):
    # Rows: Methods
    # Columns: Problems, plus Rank sum
    # Inner cells: Rank sum for each pair of problem and method, summed over all metrics
    
    summed_ranks = sum_all_ranks(methods, results_dict)
    min_value = min(summed_ranks.values())
    
    latex_table = ""
    latex_table += "\\begin{tabular}{|l|"
    for problem in problems_listed:
        latex_table += "c|"
    latex_table += "c|}\n"
    latex_table += "\\hline\n"
    latex_table += "Method & "
    for problem in problems_listed:
        latex_table += f"{problem_to_nice_problem[problem]} & "
    latex_table += "\\textbf{Rank sum} \\\\\n"
    latex_table += "\\hline\n"
    for method in methods:
        latex_table += f"{method_to_nice_method[method]} & "
        for problem in problems_listed:
            ranks = sum_ranks(problem, results_dict)
            min_rank = min(ranks.values())
            rank = ranks[method]
            if rank == min_rank and rank == summed_ranks[method]:
                latex_table += f"\\textbf{{{rank}}} & "
            elif rank == min_rank:
                latex_table += f"\\textbf{{{rank}}} & "
            elif rank == summed_ranks[method]:
                latex_table += f"\\textbf{{{rank}}} & "
            else:
                latex_table += f"{rank} & "
        if summed_ranks[method] == min_value:
            latex_table += f"\\textbf{{{summed_ranks[method]}}} \\\\\n"
        else:
            latex_table += f"{summed_ranks[method]} \\\\\n"
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}"
    
    with open(out_txt_file, "w") as f:
        f.write(latex_table)

from operator import itemgetter
def write_latex_table_content(problems_listed, methods, results_dict, out_txt_file, cell="rank", withsdegan=False):
    # Same as above, except that results_dict only has the results for one metric,
    # therefore if cell=="value", we can also print the actual value instead of a rank sum
    only_key = list(results_dict[list(results_dict.keys())[0]].keys())[0]
    summed_ranks = sum_all_ranks(methods, results_dict)
    min_value = min(summed_ranks.values())
    
    latex_table = ""
    latex_table += "\\begin{tabular}{|l|"
    for problem in problems_listed:
        latex_table += "c|"
    latex_table += "c|}\n"
    latex_table += "\\hline\n"
    latex_table += "Method & "
    for problem in problems_listed:
        latex_table += f"{problem_to_nice_problem[problem]} & "
    latex_table += "\\textbf{Rank sum} \\\\\n"
    latex_table += "\\hline\n"
    
    for method in methods:
        latex_table += f"{method_to_nice_method[method]} & "
        for problem in problems_listed:
            ranks = sum_ranks(problem, methods, results_dict)
            min_rank = min(ranks.values())
            rank = ranks[method]
            value = results_dict[(problem, method)][only_key]
            
            value_string = f"{value:.3f}" if value < 1000 else f"{value:.3e}"
            
            if rank == min_rank and rank == summed_ranks[method]:
                latex_table += f"\\textbf{{{rank}}} & " if cell=="rank" else f"\\textbf{{{value_string}}} & "
            elif rank == min_rank:
                latex_table += f"\\textbf{{{rank}}} & " if cell=="rank" else f"\\textbf{{{value_string}}} & "
            elif rank == summed_ranks[method]:
                latex_table += f"\\textbf{{{rank}}} & " if cell=="rank" else f"\\textbf{{{value_string}}} & "
            else:
                latex_table += f"{rank} & " if cell=="rank" else f"{value_string} & "
        if summed_ranks[method] == min_value:
            latex_table += f"\\textbf{{{summed_ranks[method]}}} \\\\\n"
        else:
            latex_table += f"{summed_ranks[method]} \\\\\n"
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}"
    
    with open(out_txt_file, "w") as f:
        f.write(latex_table)