import torch
import torch.utils.data as Data

from self_data_set import MyDataSet

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is shorter than time steps
sentences = [
    # enc_input           dec_input         dec_output
    ["ich mochte ein bier P", "S i want a beer .", "i want a beer . E"],
    ["ich mochte ein cola P", "S i want a coke .", "i want a coke . E"],
]

# Padding Should be Zero
src_vocab = {"P": 0, "ich": 1, "mochte": 2, "ein": 3, "bier": 4, "cola": 5}
src_vocab_size = len(src_vocab)

tgt_vocab = {
    "P": 0,
    "i": 1,
    "want": 2,
    "a": 3,
    "beer": 4,
    "coke": 5,
    "S": 6,
    "E": 7,
    ".": 8,
}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 5  # enc_input max sequence length
tgt_len = 6  # dec_input(=dec_output) max sequence length


def make_data(sentence):
    """
    构建数据集
    :param sentence:
    :return:
    """
    enc_input, dec_input, dec_output = [], [], []
    for i in range(len(sentence)):
        enc_input = [
            [src_vocab[n] for n in sentence[i][0].split()]
        ]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        dec_input = [
            [tgt_vocab[n] for n in sentence[i][1].split()]
        ]  # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_output = [
            [tgt_vocab[n] for n in sentence[i][2].split()]
        ]  # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]

        enc_input.extend(enc_input)
        dec_input.extend(dec_input)
        dec_output.extend(dec_output)

    return (
        torch.LongTensor(enc_input),
        torch.LongTensor(dec_input),
        torch.LongTensor(dec_output),
    )


if __name__ == "__main__":
    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
    for enc_inputs, dec_inputs, dec_outputs in loader:
        print("encode-input\n", enc_inputs)
        print("decode-input\n", dec_inputs)
        print("decode-output\n", dec_outputs)
