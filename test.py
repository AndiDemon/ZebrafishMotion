import torch.utils.data
from torch.backends import cudnn
# from torch.optim import SGD, Adam
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt

from utils.dataset import *
from models.rnn import *
from models.attention import *
from models.Poolformer import *

from cnn.metrics import cmat_f1, cmat_accuracy, cmat_recall, cmat_specificity

import warnings
import seaborn as sns
from ptflops import get_model_complexity_info

warnings.simplefilter('ignore')

# Set CUDA device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True


def main():
    # no_gaps()
    # data_root = Path("~/workspace/zebrafish/_data/").expanduser()
    """"
    There are 2 dataset provided:
    1. 20220527 -> taken last year May
    2. 20220921 -> taken last year September
    """
    EPOCH = 100
    data_root = Path("/net/nfs2/export/home/andi/Documents/Zebrafish/dataset/20220527/idTracker/")
    data_root_new = Path("/net/nfs2/export/home/andi/Documents/Zebrafish/dataset/20220921/convert/")

    print("TEST NEW DATA = ", len(TrajectoriesData([
        ((data_root / "old_01_trajectories_nogaps.txt").expanduser(), 1),
        ((data_root / "old_02_trajectories_nogaps.txt").expanduser(), 1),
        ((data_root / "waka_01_trajectories_nogaps.txt").expanduser(), 0),
        ((data_root / "waka_02_trajectories_nogaps.txt").expanduser(), 0),
        # ((data_root_new / "liang_y1.txt").expanduser(), 0),
        # ((data_root_new / "liang_y2.txt").expanduser(), 0),
        # ((data_root_new / "liang_o1.txt").expanduser(), 1),
        # ((data_root_new / "liang_o2.txt").expanduser(), 1),
    ])))

    valid_loader = torch.utils.data.DataLoader(
        TrajectoriesData([
            # ((data_root / "waka_01_trajectories_nogaps.txt").expanduser(), 0),
            ((data_root / "waka_02_trajectories_nogaps.txt").expanduser(), 0),
            ((data_root / "old_01_trajectories_nogaps.txt").expanduser(), 1),
            # ((data_root / "old_02_trajectories_nogaps.txt").expanduser(), 1),
            # ((data_root_new / "liang_y1.txt").expanduser(), 0),
            # ((data_root_new / "liang_y2.txt").expanduser(), 0),
            # ((data_root_new / "liang_o1.txt").expanduser(), 1),
            # ((data_root_new / "liang_o2.txt").expanduser(), 1),
        ]),
        batch_size=64, shuffle=False, num_workers=os.cpu_count() // 2
    )

    """
    x_train = np.array(x_train)[:, :, np.newaxis]
    y_train = np.array(y_train)[:, np.newaxis]
    """
    fpr_list, tpr_list, auc_list = [], [], []
    """
    Choose Model to use
    """
    model_name_list = ["GRU", "LSTM", "Poolformer", "TS-TSSA_4096", "TS-TSSA_1024", "TS-TSSA_256"]  #
    # model_name = "SelfAttention"  # GRU, Poolformer, SelfAttention, LSTM
    criterion = nn.BCELoss()

    for model_name in model_name_list:
        y_true_list, y_pred_list = [], []
        fpr_arr, tpr_arr, auc_arr = [], [], []
        model_dirname = "./checkpoints/" + model_name + ".pt"
        model = torch.load(model_dirname, map_location=device)

        total_params = sum(p.numel() for p in model.parameters())
        print("Total parameters in the model:", total_params)

        global flops
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            cmat = np.zeros(shape=(2, 2))
            for batch, (x, y_true) in enumerate(valid_loader):
                # print("x shape = ", x.shape)

                y_pred = model(x.to(device))
                y_pred = y_pred.to("cpu")

                macs, params = get_model_complexity_info(model, (300, 8), as_strings=False, print_per_layer_stat=False,
                                                         verbose=True)
                loss = criterion(y_pred, y_true)

                # print("y pred shape = ", np.argmax(y_pred, axis=1).shape)
                # print("y pred shape = ", y_pred.shape[0])
                # print(y_pred.sa)
                # for ROC-AUC Calculation
                # y_pred_list.extend(y_pred[:, np.argmax(y_pred, axis=1)])
                # y_true_list.extend(y_pred[:, np.argmax(y_true, axis=1)])

                # binary_predictions = [1 if prediction[1] >= 0.5 else 0 for prediction in one_hot_predictions]

                for i in range(y_pred.shape[0]):
                    y_pred_list.append(y_pred[i, 1])
                    y_true_list.append(np.argmax(y_true[i]).item())
                    # fpr, tpr, threshold = roc_curve(np.argmax(y_true[i]).item(), np.argmax(y_pred[i]).item())
                    # fpr_arr.append(fpr)
                    # tpr_arr.append(tpr)

                valid_loss += loss.item() / len(valid_loader)

                cmat += confusion_matrix(
                    torch.argmax(y_true, dim=1),
                    torch.argmax(y_pred, dim=1)
                )

            # print("true list = ", np.array(y_true_list).shape)
            # print("true list = ", np.array(y_true_list))
            # print("pred list = ", np.array(y_pred_list).shape)
            # print("pred list = ", np.array(y_pred_list))
            # print("pred list = ", np.array(y_pred_list).shapes)
            # Calculate the ROC Curve
            # RocCurveDisplay.from_predictions(np.array(y_true_list), np.array(y_pred_list))
            # plt.savefig('./eval/roc/test.png', bbox_inches='tight')

            fpr, tpr, threshold = roc_curve(y_true_list, y_pred_list)
            # print("tpr = ", tpr, ", fpr = ", fpr)
            # print("tpr = ", tpr[0].shape, ", fpr = ", fpr[0].shape)
            # print("pred list = ", np.array(y_pred_list).shapea)
            fpr_list.append(fpr)
            tpr_list.append(tpr)

            # Calcualte the AUC Score
            roc_auc = auc(fpr, tpr)
            auc_list.append(roc_auc)

            valid_f1 = cmat_f1(cmat)
            valid_acc = cmat_accuracy(cmat)
            valid_recall = cmat_recall(cmat)
            print(
                "\tValid Loss={:.3}, acc={:.3}, f1={:.3}, recall={:.3}, spec={:.3})".format(
                    valid_loss, cmat_accuracy(cmat), cmat_f1(cmat),
                    cmat_recall(cmat), cmat_specificity(cmat)
                ), end=""
            )
            print("")
            print("valid f1 = ", valid_f1, ", best acc = ", valid_acc, ", best recall = ", valid_recall)
            print("FLOPS:", int(macs) * 2)
            # print("FLOPS:", int(flops))
            """
            save confusion matrix
            """
            # ax = sns.heatmap(cmat, linewidths=1, annot=True, fmt='g')
            #
            # ax.set_xlabel("Prediction", fontsize=14, labelpad=20)
            # ax.xaxis.set_ticklabels(['Young', 'Old'])
            #
            # ax.set_ylabel("True", fontsize=14, labelpad=20)
            # ax.yaxis.set_ticklabels(['Young', 'Old'])
            #
            # # ax.set_title("Confusion Matrix by LSTM Model", fontsize=14, pad=20)
            # plt.savefig('./eval/confusion_matrix/eps/' + model_name + '.eps', bbox_inches='tight')

        print("pred list = ", np.array(y_pred_list).shape)

    """
    save ROC Curve
    """
    colors = ['b--', 'g:', 'r-.', 'c-', 'm-', 'y-']
    plt.title('ROC Curve')
    for i in range(len(model_name_list)):
        plt.plot(fpr_list[i], tpr_list[i], colors[i], label='AUC ' + model_name_list[i] + ' = %0.2f' % auc_list[i])
    plt.legend(loc='lower right', fontsize='8')
    # plt.plot([0, 1], [0, 1], 'k--', alpha=.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('./eval/roc/eps/test.eps', bbox_inches='tight')


if __name__ == '__main__':
    main()
