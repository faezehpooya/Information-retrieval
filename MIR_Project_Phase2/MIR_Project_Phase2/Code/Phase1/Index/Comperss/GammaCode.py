class GammaCodeCompressor:

    @staticmethod
    def write_compress_to_file(index, file_name='GammaCode_index.txt'):
        file = open(file_name, "wb")
        
        sorted_index = sorted(index)
        file.write(int(len(str(sorted_index).encode())).to_bytes(4, 'little'))
        file.write(str(sorted_index).encode())
        for t_id in sorted_index:
            posting_dict = index[t_id]
            doc_id_list = sorted(posting_dict)
            gamma_code = GammaCodeCompressor.compress_posting_list(doc_id_list)
            GammaCodeCompressor.write_compressed_list(file, gamma_code)
            for doc_id in doc_id_list:
                if posting_dict[doc_id].get('title') is not None:
                    file.write(int(1).to_bytes(1, 'little'))
                    gamma_code = GammaCodeCompressor.compress_posting_list(posting_dict[doc_id]['title'])
                    GammaCodeCompressor.write_compressed_list(file, gamma_code)
                else:
                    file.write(int(0).to_bytes(1, 'little'))
                    
                if posting_dict[doc_id].get('description') is not None:
                    file.write(int(1).to_bytes(1, 'little'))
                    gamma_code = GammaCodeCompressor.compress_posting_list(posting_dict[doc_id]['description'])
                    GammaCodeCompressor.write_compressed_list(file, gamma_code)
                else:
                    file.write(int(0).to_bytes(1, 'little'))
        file.close()

    @staticmethod
    def write_compressed_list(file, gamma_code):
        gamma_code = '0' * (8 - (len(gamma_code) % 8)) + gamma_code
        file.write(int(len(gamma_code)/8).to_bytes(4, 'little'))
        for i in range(0, len(gamma_code), 8):
            file.write(int(gamma_code[i:i + 8], 2).to_bytes(1, 'little'))

    @staticmethod
    def compress_posting_list(posting_list):
        gamma_code = ''
        if not len(posting_list):
            return gamma_code
        gamma_code += GammaCodeCompressor.gamma_code(posting_list[0] + 1)
        for i in range(1, len(posting_list)):
            gamma_code += GammaCodeCompressor.gamma_code(posting_list[i] - posting_list[i - 1])
        return gamma_code

    @staticmethod
    def gamma_code(num):
        offset = format(num, 'b')[1:]
        unary = '1' * len(offset) + '0'
        return unary + offset


class GammaCodeDecompressor:

    @staticmethod
    def decompress_from_file(file_name='GammaCode_index.txt'):
        file = open(file_name, "rb")
        index = {}
        t_id_length = int(format(int.from_bytes(file.read(4), 'little')))
        t_id = eval(file.read(t_id_length).decode())
        for i in t_id:
            posting_dict = {}
            doc_list = GammaCodeDecompressor.read_compressed_list(file)
            for doc_id in doc_list:
                poses = {}
                have_title_poses = int(format(int.from_bytes(file.read(1), 'little')))
                if have_title_poses:
                    title_poses = GammaCodeDecompressor.read_compressed_list(file)
                    poses['title'] = title_poses
                
                have_description_poses = int(format(int.from_bytes(file.read(1), 'little')))
                if have_description_poses:
                    description_poses = GammaCodeDecompressor.read_compressed_list(file)
                    poses['description'] = description_poses   
                posting_dict[doc_id] = poses
            index[i] = posting_dict
        file.close()
        return index

    @staticmethod
    def read_compressed_list(file):
        length = int(format(int.from_bytes(file.read(4), 'little')))
        posting_byte = file.read(length)
        gamma_code = ''
        for byte in posting_byte:
            gamma_code += format(byte, '08b')
        gamma_code = gamma_code.lstrip('0')
        if not gamma_code:
            return [0]
        return GammaCodeDecompressor.decompress_posting_list(gamma_code)

    @staticmethod
    def decompress_posting_list(gamma_code):
        posting_list = []
        prev_num = -1
        while gamma_code != "":
            unary_len = gamma_code.find('0')
            offset = '1' + gamma_code[unary_len + 1: 2 * unary_len + 1]
            num = int(offset, 2) + prev_num
            posting_list.append(num)
            prev_num = num
            gamma_code = gamma_code[2 * unary_len + 1:]
        return posting_list
