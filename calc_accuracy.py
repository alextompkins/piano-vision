def main(song_name):
	with open('ground_truths/{}'.format(song_name)) as ground_truth_file:
		with open('output/{}.log'.format(song_name)) as output_file:
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

			print('Correct: {}'.format(correct))
			print('False Negatives: {}'.format(false_negative))
			print('False Positives: {}'.format(false_positive))
			print('Precision: {:.2f}%'.format(correct / (correct + false_negative) * 100))
			print('Recall: {:.2f}%'.format(correct / (correct + false_positive) * 100))


if __name__ == '__main__':
	main('canon_in_d')
