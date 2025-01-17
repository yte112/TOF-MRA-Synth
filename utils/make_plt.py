import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def excel_dict(root, keys, mode, save_data=None):
    if mode == 'read':
        data = pd.read_excel(root)
        excel_dict = dict(zip(keys, [list(data[key]) for key in keys]))
        return excel_dict
    else:
        df = pd.DataFrame(save_data)  # 创建DataFrame
        df.to_excel(root, index=False)  # 


def draw_sub(location, l, key, train_dict, val_dict):
    x = np.linspace(1, l, l)
    l_trian = train_dict[key]
    l_val = val_dict[key]
    plt.subplot(*location)
    plt.title(key)
    plt.xlabel('train times')
    plt.ylabel(key)
    plt.plot(x, l_trian, label='train')
    plt.plot(x, l_val, label='val')
    plt.legend()


def main():
    keys = ['MIP_LOSS', 'PSNR', 'SSIM', 'SNR']
    train_dict = excel_dict('./logs/train.xlsx', keys, 'read')
    val_dict = excel_dict('./logs/val.xlsx', keys, 'read')
    l_ = len(val_dict[keys[0]])
    print(l_)

    draw_sub([2, 2, 1], l_, keys[0], train_dict, val_dict)
    draw_sub([2, 2, 2], l_, keys[1], train_dict, val_dict)
    draw_sub([2, 2, 3], l_, keys[2], train_dict, val_dict)
    draw_sub([2, 2, 4], l_, keys[3], train_dict, val_dict)

    plt.tight_layout(pad=2)
    plt.savefig('progress.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
