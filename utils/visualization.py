import matplotlib.pyplot as plt
import numpy as np


def plot(x, y, y2):
    plt.plot(x, y2, label='MedRetriever', linestyle='-', c='blue')
    plt.plot(x, y, label='RetainEx', linestyle='--', c='red')
    plt.xlabel('memory size', fontsize=13)
    plt.ylabel('AUC', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlim(-0.1, 4.1)
    plt.ylim(0.5, 0.85)
    plt.legend()
    plt.show()


def bar(x, y, y2):
    plt.bar(x=x, height=y2, label='With Text', color=['indianred', 'indianred', 'indianred', 'indianred', 'indianred', 'indianred', 'indianred', 'indianred', 'indianred'], alpha=0.5)
    plt.bar(x=x, height=y, label='Vanilla', color='steelblue', alpha=0.5)
    for x, (y, y2) in enumerate(zip(y, y2)):
        if y <= y2:
            plt.text(x, y - 0.001, y, ha='center', va='top')
            plt.text(x, y2 + 0.001, y2, ha='center', va='bottom')
        else:
            plt.text(x, y2 - 0.001, y2, ha='center', va='top')
            plt.text(x, y + 0.001, y, ha='center', va='bottom')
    plt.xticks(rotation=35)
    plt.xlim(-1, 9)
    plt.ylim(0.64, 0.84)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.xlabel("EHR Encoder", fontsize=13)
    plt.ylabel("AUC", fontsize=13)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # hf_y2 = np.array([0.754, 0.746, 0.750, 0.746, 0.751])
    # hf_y = np.array([0.708, 0.708, 0.708, 0.708, 0.708])
    # x = ['5', '10', '15', '20', '25']
    # plot(x, hf_y, hf_y2)
    #
    # copd_y2 = np.array([0.728, 0.740, 0.739, 0.718, 0.739])
    # copd_y = np.array([0.693, 0.693, 0.693, 0.693, 0.693])
    # plot(x, copd_y, copd_y2)
    #
    # kidney_y2 = np.array([0.778, 0.778, 0.775, 0.759, 0.782])
    # kidney_y = np.array([0.739, 0.739, 0.739, 0.739, 0.739])
    # plot(x, kidney_y, kidney_y2)
    #
    # amniesia_y2 = np.array([0.767, 0.782, 0.772, 0.773, 0.772])
    # amniesia_y = np.array([0.694, 0.694, 0.694, 0.694, 0.694])
    # plot(x, amniesia_y, amniesia_y2)
    #
    # dementia_y2 = np.array([0.755, 0.739, 0.745, 0.746, 0.725])
    # dementia_y = np.array([0.713, 0.713, 0.713, 0.713, 0.713])
    # plot(x, dementia_y, dementia_y2)

    #retainex
    hf_y2 = np.array([0.785, 0.779, 0.776, 0.784, 0.780])
    hf_y = np.array([0.688, 0.688, 0.688, 0.688, 0.688])
    x = ['5', '10', '15', '20', '25']
    plot(x, hf_y, hf_y2)

    copd_y2 = np.array([0.779, 0.780, 0.779, 0.780, 0.760])
    copd_y = np.array([0.707, 0.707, 0.707, 0.707, 0.707])
    plot(x, copd_y, copd_y2)

    kidney_y2 = np.array([0.811, 0.799, 0.799, 0.813, 0.803])
    kidney_y = np.array([0.728, 0.728, 0.728, 0.728, 0.728])
    plot(x, kidney_y, kidney_y2)

    #bar
    x = ['LSTM', 'Dipole', 'Retain', 'SAnD', 'LSAN', 'RetainEx', 'Timeline', 'HiTANet', 'GRAM']
    hf_y = [0.708, 0.687, 0.689, 0.686, 0.738, 0.688, 0.705, 0.750, 0.748]
    hf_y2 = [0.746, 0.762, 0.727, 0.708, 0.743, 0.773, 0.740, 0.756, 0.754]
    bar(x, hf_y, hf_y2)

    copd_y = [0.693, 0.704, 0.699, 0.692, 0.723, 0.707, 0.698, 0.752, 0.722]
    copd_y2 = [0.718, 0.734, 0.716, 0.700, 0.736, 0.777, 0.721, 0.752, 0.752]
    bar(x, copd_y, copd_y2)
    kidney_y = [0.739, 0.755, 0.732, 0.748, 0.766, 0.728, 0.756, 0.792, 0.780]
    kidney_y2 = [0.759, 0.770, 0.756, 0.744, 0.785, 0.802, 0.754, 0.796, 0.793]
    bar(x, kidney_y, kidney_y2)
    #
    # amniesia_y = [0.694, 0.723, 0.739, 0.735, 0.710, 0.744, 0.745, 0.762, 0.792]
    # amniesia_y2 = [0.773, 0.759, 0.741, 0.759, 0.791, 0.759, 0.784, 0.783, 0.788]
    # bar(x, amniesia_y, amniesia_y2, 'Amnesia')
    #
    # dementia_y = [0.713, 0.673, 0.692, 0.611, 0.690, 0.664, 0.720, 0.723, 0.728]
    # dementia_y2 = [0.746, 0.724, 0.728, 0.675, 0.743, 0.737, 0.725, 0.720, 0.726]
    # bar(x, dementia_y, dementia_y2, 'Dementia')




