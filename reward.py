import re
from random import randint, seed, choice
from typing import List, Tuple


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = set(available_numbers)
        numbers_in_eq = set(numbers_in_eq)
        
        return numbers_in_eq.issubset(available_numbers)
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def get_reward(solution_str, target, numbers, format_score=0.1, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    
    equation = extract_solution(solution_str=solution_str)
    do_print = randint(1, 256) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


def get_rewards(samples, targets, numbers) -> List[float]:
    assert(len(samples) == len(targets) == len(numbers))
    return list(map(get_reward, samples, targets, numbers))


def get_sample_stats(rewards):
    ratio_correct = sum([r == 1 for r in rewards]) / len(rewards)
    ratio_incorrect_valid_format = sum([r == 0.1 for r in rewards]) / len(rewards)
    ratio_incorrect_invalid_format = sum([r == 0 for r in rewards]) / len(rewards)
    return {"ratio_correct": ratio_correct,
            "ratio_incorrect_valid_format": ratio_incorrect_valid_format,
            "ratio_incorrect_invalid_format": ratio_incorrect_invalid_format}
    