import os

PROCESSING_PATH= os.path.dirname(os.path.abspath(__file__))
MOVERS_PATH1 = os.path.join(PROCESSING_PATH, 'data/alistair/mover.csv')
POSITION_PATH = os.path.join(PROCESSING_PATH, 'data/csv/all_movers/position.csv')
BIG_IMAGE_PATH = os.path.join(PROCESSING_PATH, 'data/alistair/images/')
SMALL_IMAGE_PATH = os.path.join(PROCESSING_PATH, 'data/alistair/30x30_images/') 
POS_MOVER_PATH = os.path.join(PROCESSING_PATH, 'data/alistair/filtered_metadata_pos.csv')
NEG_MOVER_PATH = os.path.join(PROCESSING_PATH, 'data/alistair/filtered_metadata_neg.csv')
