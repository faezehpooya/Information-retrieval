class VariableByteCompressor:

    @staticmethod
    def write_compress_to_file(index, file_name='VariableByte_index.txt'):
        file = open(file_name, "wb")

        sorted_index = sorted(index)
        file.write(int(len(str(sorted_index).encode())).to_bytes(4, 'little'))
        file.write(str(sorted_index).encode())
        for t_id in sorted_index:
            posting_dict = index[t_id]
            doc_id_list = sorted(posting_dict)
            variable_byte = VariableByteCompressor.compress_posting_list(doc_id_list)
            VariableByteCompressor.write_compressed_list(file, variable_byte)
            for doc_id in doc_id_list:
                if posting_dict[doc_id].get('title') is not None:
                    file.write(int(1).to_bytes(1, 'little'))
                    variable_byte = VariableByteCompressor.compress_posting_list(posting_dict[doc_id]['title'])
                    VariableByteCompressor.write_compressed_list(file, variable_byte)
                else:
                    file.write(int(0).to_bytes(1, 'little'))
                    
                if posting_dict[doc_id].get('description') is not None:
                    file.write(int(1).to_bytes(1, 'little'))
                    variable_byte = VariableByteCompressor.compress_posting_list(posting_dict[doc_id]['description'])
                    VariableByteCompressor.write_compressed_list(file, variable_byte)
                else:
                    file.write(int(0).to_bytes(1, 'little'))
        file.close()

    @staticmethod
    def write_compressed_list(file, variable_byte):
        file.write(int(len(variable_byte)).to_bytes(4, 'little'))
        for byte in variable_byte:
            file.write(byte)
            file.flush()

    @staticmethod
    def compress_posting_list(posting_list):
        variable_byte = []
        if not len(posting_list):
            return variable_byte
        variable_byte += VariableByteCompressor.variable_byte(posting_list[0])
        for i in range(1, len(posting_list)):
            variable_byte += VariableByteCompressor.variable_byte(posting_list[i] - posting_list[i - 1])
        return variable_byte

    @staticmethod
    def variable_byte(num):
        binary_str = format(num, 'b')
        remaining = binary_str
        variable_byte = []
        end_bit = '1'
        for i in range(len(binary_str), 7, -7):
            byte = end_bit + binary_str[i - 7: i]
            variable_byte.insert(0, int(byte, 2).to_bytes(1, 'little'))
            remaining = binary_str[:i - 7]
            end_bit = '0'
        if remaining:
            byte = end_bit + '0' * (7 - len(remaining)) + remaining
            variable_byte.insert(0, int(byte, 2).to_bytes(1, 'little'))
        return variable_byte


class VariableByteDecompressor:

    @staticmethod
    def decompress_from_file(file_name='VariableByte_index.txt'):
        file = open(file_name, "rb")
        index = {}
        t_id_length = int(format(int.from_bytes(file.read(4), 'little')))
        t_id = eval(file.read(t_id_length).decode())
        for i in t_id:
            posting_dict = {}
            doc_list = VariableByteDecompressor.read_compressed_list(file)
            for doc_id in doc_list:
                poses = {}
                have_title_poses = int(format(int.from_bytes(file.read(1), 'little')))
                if have_title_poses:
                    title_poses = VariableByteDecompressor.read_compressed_list(file)
                    poses['title'] = title_poses
                
                have_description_poses = int(format(int.from_bytes(file.read(1), 'little')))
                if have_description_poses:
                    description_poses = VariableByteDecompressor.read_compressed_list(file)
                    poses['description'] = description_poses   
                posting_dict[doc_id] = poses
            index[i] = posting_dict
        file.close()
        return index

    @staticmethod
    def read_compressed_list(file):
        length = int(format(int.from_bytes(file.read(4), 'little')))
        posting_byte = file.read(length)
        variable_byte = []
        for byte in posting_byte:
            variable_byte.append(format(byte, '08b'))
        return VariableByteDecompressor.decompress_posting_list(variable_byte)

    @staticmethod
    def decompress_posting_list(variable_byte):
        posting_list = []
        prev_num = 0
        num = ''
        i = 0
        while i < len(variable_byte):
            while variable_byte[i][0] != '1':
                num += variable_byte[i][1:]
                i += 1
            num += variable_byte[i][1:]
            i += 1
            prev_num += int(num, 2)
            posting_list.append(prev_num)
            num = ''
        return posting_list
