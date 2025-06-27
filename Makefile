all: data_preprocessing gan train_simclr train_classifier_vit train_classifier_mlp_mixer train_classifier_cnn

data_preprocessing:
	python3 data_preprocessing.py ../data/original ../data/preprocessed ../data/original/UCSF-PDGM-metadata_v2.csv

gan: ../data/preprocessed/all_grades_df.csv
	python3 gan.py ../data/preprocessed/all_grades_df.csv ../data/gan_results

train_simclr: ../data/gan_results/original_and_gan_df.csv
	python3 train_simclr.py ../data/gan_results/original_and_gan_df.csv ../data/simclr_results

train_classifier_vit: ../data/simclr_results/simclr_encoder.pth
	python3 train_classifier.py ../data/gan_results/original_and_gan_df.csv ../data/simclr_results/simclr_encoder.pth ../data/classifier_results vit --num_epochs 5

train_classifier_mlp_mixer: ../data/simclr_results/simclr_encoder.pth
	python3 train_classifier.py ../data/gan_results/original_and_gan_df.csv ../data/simclr_results/simclr_encoder.pth ../data/classifier_results mlp_mixer

train_classifier_cnn: ../data/simclr_results/simclr_encoder.pth
	python3 train_classifier.py ../data/gan_results/original_and_gan_df.csv ../data/simclr_results/simclr_encoder.pth ../data/classifier_results cnn

requirements:
	pip-compile --max-rounds=5 requirements.in

clean:
	rm -rfv ../data/preprocessed
	rm -rfv ../data/gan_results
	rm -rfv ../data/simclr_results
	rm -rfv ../data/classifier_results
