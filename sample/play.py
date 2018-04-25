import tensorflow as tf
import tensorflow_hub as hub
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
embeddings = embed([
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"])

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    print(session.run(embeddings))