import heapq


class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(char_freq):
    # Create a priority queue to store Huffman nodes
    min_heap = []
    for char, freq in char_freq.items():
        node = HuffmanNode(char, freq)
        heapq.heappush(min_heap, node)

    # Build the Huffman tree
    while len(min_heap) > 1:
        left = heapq.heappop(min_heap)
        right = heapq.heappop(min_heap)
        parent = HuffmanNode(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        heapq.heappush(min_heap, parent)

    return min_heap[0]  # The root of the Huffman tree


def build_huffman_codes(root, current_code, huffman_codes):
    if root is None:
        return

    if root.char is not None:
        huffman_codes[root.char] = current_code
        return

    build_huffman_codes(root.left, current_code + '0', huffman_codes)
    build_huffman_codes(root.right, current_code + '1', huffman_codes)


def huffman_encoding(data):
    char_freq = {}
    for char in data:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1

    # Build the Huffman tree
    root = build_huffman_tree(char_freq)

    # Build the Huffman codes
    huffman_codes = {}
    build_huffman_codes(root, '', huffman_codes)

    encoded_data = ''
    for char in data:
        encoded_data += huffman_codes[char]

    return encoded_data, root


def huffman_decoding(encoded_data, root):
    decoded_data = ''
    current_node = root

    for bit in encoded_data:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.char is not None:
            decoded_data += current_node.char
            current_node = root

    return decoded_data


# Example usage:
if __name__ == "__main":
    data = "this is an example for huffman encoding"

    encoded_data, tree = huffman_encoding(data)
    print(f"Encoded data: {encoded_data}")

    decoded_data = huffman_decoding(encoded_data, tree)
    print(f"Decoded data: {decoded_data}")