import sys


def main(song, log):
	with open('ground_truths/{}'.format(song)) as ground_truth_file:
		with open(log) as output_file:
			correct = 0
			false_positive = 0
			false_negative = 0
			total_truths = 0

			truths = ground_truth_file.readlines()
			outputs = output_file.readlines()

			for i, truth in enumerate(truths):
				truth = set(truth[truth.find('[') + 1:truth.find(']')].split(', '))
				output = set(outputs[i][outputs[i].find('[') + 1:outputs[i].find(']')].split(', '))

				correct += len(truth.intersection(output))
				false_positive += len(output.difference(truth))
				false_negative += len(truth.difference(output))
				total_truths += len(truth)

			precision = correct / (correct + false_negative)
			recall = correct / (correct + false_positive)
			f1 = 2 * (precision * recall) / (precision + recall)
			print('Correct: {}'.format(correct))
			print('False Negatives: {}'.format(false_negative))
			print('False Positives: {}'.format(false_positive))
			print('Precision: {:.2f}%'.format(precision * 100))
			print('Recall: {:.2f}%'.format(recall * 100))
			print('F1 score: {:.3f}'.format(f1))


if __name__ == '__main__':
	if len(sys.argv) <= 1:
		print('Usage: calc_accuracy.py song_name [log_name]')
		sys.exit(1)
	song_name = sys.argv[1]

	if len(sys.argv) >= 3:
		log_name = 'output/{}-{}.log'.format(song_name, sys.argv[2])
	else:
		log_name = 'output/{}.log'.format(song_name)

	main(song_name, log_name)
