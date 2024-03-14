import json
import re
import os

from metrics import bleu_score, chrfplus_score, cter_score, em_score
from util import split_bidirectional, is_directional
from nltk.tokenize import word_tokenize
import random


##############################
####### Preprocessing ########
##############################

def remove_punctuation(sent):
    """
    Remove fullstops (.), exclamation marks (!), question marks (?), and commas (,) from a sentence
    :param sent: ground truth or submission sentence that may include punctuation
    :return: ground truth or submission sentence without punctuation
    """
    return re.sub(r"[.!?,]","",sent)

def replace_brackets(sent):
    """
    Replace brackets "[word]" with "(word/ )" in the sentence
    :param sent: ground truth or submission sentence that may include brackets
    :return: ground truth or submission sentence without brackets
    """
    return re.sub("\[([^]]+)\]", '(\\1/ )', sent)

def get_alternatives(gt_sent, alternatives):
    """
    Get all valid alternative writings of a ground truth sentence (recursive function)
    :param gt_sent: ground truth sentence
    :param alternatives: empty list to start recursive loop
    :return: list of all valid alternative writings of gt_sent
    """
    alternatives.append(gt_sent)
    if (len(re.findall(r"\([^\)\r\n]+\/[^\)\r\n]+\)", gt_sent)) > 0):
        # get first parenthesis
        parenthesis = re.findall(r"\([^\)\r\n]+\/[^\)\r\n]+\)", gt_sent)[0]
        # get all options provided within the parenthesis
        options = [i.strip() for i in parenthesis[1:-1].split("/")]
        # recursively loop through each option
        for o in options:
            get_alternatives(gt_sent.replace(parenthesis, o), alternatives)
    #else:
    #    return alternatives
    return alternatives

def expand_options(text):
    """Expands options enclosed in parentheses and separated by slashes into all possible combinations."""
    def split_options(option_text):
        """Split options within parentheses, avoiding nested ones."""
        return option_text.split('/')

    def recurse(text, start=0, current_combo=[]):
        if start >= len(text):
            # Join the current combination back into a string and append to results
            results.append(''.join(current_combo))
            return

        # Look for the next set of parentheses
        next_parentheses = re.search(r'\(([^()]+)\)', text[start:])
        if not next_parentheses:
            # No more parentheses, add the remainder of the text
            recurse(text, len(text), current_combo + [text[start:]])
        else:
            # Process content within parentheses
            before, options, after = text[start:next_parentheses.start()], next_parentheses.group(1), text[next_parentheses.end():]
            for option in split_options(options):
                recurse(after, 0, current_combo + [before, option])

    results = []
    recurse(text)
    return results

def remove_pronoun_tags(sent):
    """
    Remove pronoun tags (i.e. .SG/.PL/.PL2/.PL3) from a sentence
    :param sent: ground truth or submission sentence that may include pronoun tags
    :return: ground truth or submission sentence without pronoun tags
    """
    # list of pronoun tags
    tags = [".SG", ".PL2-", ".PL2", ".PL3", ".PL"]

    # remove the tags
    for t in tags:
        sent = sent.replace(t, "")
    return sent

##########################
######  Evaluation   #####
##########################

def evaluate_puzzle(ground_truth, submission):
    """
    Preprocess the sentences, then calculate the scores
    :param ground_truth:
    :param submission:
    :return:
    """
    bleu_scores = []
    chrfpp_scores = []
    cter_scores = []
    em_scores = []

    try:
        for i in range(len(ground_truth["test"])):
            assert (ground_truth["test"][i][0].strip() == submission["test"][i][0].strip()), "PLEASE KEEP THE ORDER OF SUBMISSION SENTENCES INTACT"

            gt_sent = ground_truth["test"][i][1]
            sub_sent = submission["test"][i][1]

            ref = gt_sent.strip()
            sub = sub_sent.strip()

            ref = remove_pronoun_tags(replace_brackets(ref))
            sub = remove_pronoun_tags(replace_brackets(sub))
            ref = ref.lower()
            sub = sub.lower()
            ref = remove_punctuation(ref)
            sub = remove_punctuation(sub)

            # create alternative translations from the reference text (he/she) -> (he/she), he or she
            gt_sent_lst = get_alternatives(ref, [])
            sub_sent_lst = get_alternatives(sub, [])

            max_bleu = 0.
            max_chrf = 0.
            max_cter = 0.
            max_em = 0.

            for sub in sub_sent_lst:
                bs = bleu_score(gt_sent_lst, sub)
                if bs > max_bleu:
                    max_bleu = bs
                chrf = chrfplus_score(gt_sent_lst, sub)
                if chrf > max_chrf:
                    max_chrf = chrf
                cter = cter_score(gt_sent_lst, sub)
                if cter > max_cter:
                    max_cter = cter
                em = em_score(gt_sent_lst, sub)
                if em > max_em:
                    max_em = em

            bleu_scores.append(max_bleu)
            chrfpp_scores.append(max_chrf)
            cter_scores.append(max_cter)
            em_scores.append(max_em)

    except Exception as e:
        raise ValueError(repr(e))

    return {
        "bleu": bleu_scores,
        "chrf": chrfpp_scores,
        "cter": cter_scores,
        "em": em_scores
    }

def evaluate_unidirectional(ground_truth, submission):
    return evaluate_puzzle(ground_truth, submission)

def evaluate_directional(ground_truth, submission):
    """
    Evaluates and returns the scores for one puzzle.
    Direction is fixed, ltr = always foreign -> English and vice versa
    :param ground_truth: reference puzzle
    :param submission: submitted puzzle
    :return: scores for english->foreign, foreign->english and all
    """
    # ltr: puzzle that contains source to target test questions; rtl: puzzle that contains target to source test pairs
    #                                                               : other info reversed too

    gt_ltr, gt_rtl = split_bidirectional(ground_truth)
    pred_ltr, pred_rtl = split_bidirectional(submission)
    assert (gt_ltr is None) == (pred_ltr is None), "PLEASE KEEP TRANSLATION DIRECTIONS INTACT!"
    assert (gt_rtl is None) == (pred_rtl is None), "PLEASE KEEP TRANSLATION DIRECTIONS INTACT!"

    have_ltr = (gt_ltr is not None)
    have_rtl = (gt_rtl is not None)

    if have_ltr:
        # Evaluate source -> target
        ltr_eval = evaluate_puzzle(gt_ltr, pred_ltr)
    else:
        ltr_eval = {"bleu": [], "chrf": [], "cter": [], "em": []}
    if have_rtl:
        # Evaluate target -> source
        rtl_eval = evaluate_puzzle(gt_rtl, pred_rtl)
    else:
        rtl_eval = {"bleu": [], "chrf": [], "cter": [], "em": []}

    return (rtl_eval, ltr_eval)

def evaluate_file(gt_file, submission_file):
    """
    Evaluate one file
    :param gt_file:
    :param submission_file:
    :return:
    """
    with open(gt_file, encoding="utf8") as file:
        ground_truth = json.load(file)
    try:
        with open(submission_file, encoding="utf8") as file:
            submission = json.load(file)

    except Exception:
        print(f"ERROR: {submission_file} missing.")
        raise ValueError(f"ERROR: {submission_file} missing.")

    gt_directional = is_directional(ground_truth)
    assert is_directional(submission) == gt_directional, "PLEASE KEEP TRANSLATION DIRECTIONS INTACT!"

    if gt_directional:
        return evaluate_directional(ground_truth, submission)
    else:
        return evaluate_unidirectional(ground_truth, submission)


# my written code
def preprocess_sentence(sentence):
    sentence = replace_brackets(sentence)
    sentence = remove_pronoun_tags(sentence)
    sentence = remove_punctuation(sentence)
    sentence = sentence.lower()
    alternatives = expand_options(sentence)
    alts = []
    for alternative in alternatives:
        temp = expand_options(alternative)
        for item in temp:
            alts.append(item)
    alts = list(dict.fromkeys(alts))
    return alts

def preprocess_and_expand(sentence):
    """
    Expand sentences with alternatives like "(The/A)" into all possible combinations.
    """
    sentence = replace_brackets(sentence)
    sentence = remove_pronoun_tags(sentence)
    sentence = remove_punctuation(sentence)
    sentence = sentence.lower()
    def expand(match):
        # Split the content of the parentheses and return it as separate options
        return match.group(1).split('/')

    # Use a recursive function to handle nested alternatives
    def recursive_expand(text, start=0):
        # Find the next set of parentheses with alternatives
        match = re.search(r'\(([^()]+)\)', text)
        if not match:
            return [text]

        before = text[:match.start()]
        after = text[match.end():]
        alternatives = expand(match)

        # Generate all combinations of the sentence with the current set of alternatives
        results = []
        for alt in alternatives:
            expanded_sentences = recursive_expand(before + alt + after)
            for expanded_sentence in expanded_sentences:
                results.append(expanded_sentence)
        results = list(dict.fromkeys(results))
        return results

    # Start the expansion process
    return recursive_expand(sentence)

def random_words_evaluate_directory(ground_truth_dir):
    aggregated_scores = {"bleu": [], "chrf": [], "cter": [], "em": []}

    for filename in os.listdir(ground_truth_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(ground_truth_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Expand ground truth and generate corresponding translations
            expanded_ground_truth = []
            submission_test = []
            english_vocabulary = set()
            for _, english_sentence in data['train']:
                processed_sentence = preprocess_sentence(english_sentence)
                for sentence in processed_sentence:  # Assuming preprocess_sentence returns a list of alternatives
                    english_vocabulary.update(word_tokenize(sentence.lower()))
            for source_sentence, target_sentence, _ in data['test']:
                # Expand target sentence alternatives
                target_expansions = preprocess_and_expand(target_sentence)
                for expanded_target in target_expansions:
                    # Duplicate source sentence for each target expansion
                    expanded_ground_truth.append([source_sentence, expanded_target])

                    # Generate translation for the expanded target sentence
                    translation = ' '.join(random.choice(list(english_vocabulary)) for _ in word_tokenize(expanded_target))
                    submission_test.append([source_sentence, translation])

            # Construct submission and ground truth structures
            submission = {"source_language": data["source_language"], "target_language": data["target_language"], "test": submission_test}
            ground_truth = {"source_language": data["source_language"], "target_language": data["target_language"], "test": expanded_ground_truth}
            # Evaluate the submission against the ground truth
            evaluation_results = evaluate_unidirectional(ground_truth, submission)

            # Aggregate the scores
            for metric in aggregated_scores:
                aggregated_scores[metric].extend(evaluation_results[metric])

    # Compute average scores
    average_scores = {metric: sum(scores) / len(scores) for metric, scores in aggregated_scores.items() if scores}
    return average_scores

# Path to the directory containing your JSON files
ground_truth_dir = 'trial_data_with_answers'
average_scores = random_words_evaluate_directory(ground_truth_dir)

# Print the average scores across all evaluated files
print("Random Words Average Scores Across All Files:", json.dumps(average_scores, indent=4))


# Specify the paths to your files
ground_truth_file = 'trial_data_with_answers/a77783c1f82d55853b0be65f7402fa9995668d77ca996707de4a9b2b249acd78.json'
submission_file = 'gpt/baselineyonggom_gpt.json'

# Call the function to evaluate the submission file against the ground truth
results = evaluate_file(ground_truth_file, submission_file)
with open(submission_file, 'r', encoding='utf-8') as file:
    data = json.load(file)
file_name = data['source_language']
avg_srctotarg_score = sum(results[0]['bleu']) / len(results[0]['bleu'])
avg_chrf = sum(results[0]['chrf']) / len(results[0]['chrf'])
avg_cter = sum(results[0]['cter']) / len(results[0]['cter'])
avg_em = sum(results[0]['em']) / len(results[0]['em'])
print("Average Source to Target BLEU Score for", file_name, ":", avg_srctotarg_score)
print("Average Source to Target CHRF Scores for", file_name, ":", avg_chrf)
print("Average Source to Target CTER Scores for", file_name, ":", avg_cter)
print("Average Source to Target EM Scores for", file_name, ":", avg_em)

avg_targtosrc_score = sum(results[1]['bleu']) / len(results[1]['bleu'])
avg_chrf = sum(results[1]['chrf']) / len(results[1]['chrf'])
avg_cter = sum(results[1]['cter']) / len(results[1]['cter'])
avg_em = sum(results[1]['em']) / len(results[1]['em'])
print("Average Target to Source BLEU Score for", file_name, ":", avg_targtosrc_score)
print("Average Target to Source CHRF Scores for", file_name, ":", avg_chrf)
print("Average Target to Source CTER Scores for", file_name, ":", avg_cter)
print("Average Target to Source EM Scores for", file_name, ":", avg_em)


