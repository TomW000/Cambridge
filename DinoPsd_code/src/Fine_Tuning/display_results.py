from setup import neurotransmitters
from setup import plt, np

def training_curve(epochs, loss_list, test_accuracies):

    x = [i for i in range(epochs)]
    fig, ax1 = plt.subplots(figsize=(5,5), dpi=150)
    ax2 = ax1.twinx()
    lns1 = ax1.plot(x, loss_list, label='Train loss')
    ax1.set_ylim(0,max(loss_list)*1.05)
    lns2 = ax2.plot(x, test_accuracies, label='Test accuracy', color='red')
    ax2.set_ylim(0,105)

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    ax1.set_ylabel('Test accuracy')
    ax2.set_ylabel('Train loss', rotation=-90)
    ax1.set_xlabel('Epochs')

    plt.show()


def class_proprtions(proportion_list):

    one_hot_neurotransmitters = np.eye(len(neurotransmitters))
    
    gts = []
    for e in proportion_list: gts.extend(e)
    gts = np.array(gts)

    vectors, counts = np.unique(gts, axis=0, return_counts=True)
    positions = [np.where(np.all(one_hot_neurotransmitters == v, axis=1)) for v in vectors]

    proportions = np.zeros((len(neurotransmitters),1))
    for count, position in zip(counts, positions): proportions[position] = count

    proportions = 100*proportions/int(np.sum(proportions))

    fig, ax = plt.subplots(figsize=(10,4), dpi=150)

    img = ax.imshow(proportions.T, cmap='RdYlGn')

    for k, prop in enumerate(proportions):
        text = ax.text(x=k, y=0, s=f'{round(proportions[k].item(), ndigits=2)}%\n({int(counts[k])})', ha="center", va="center", color="black")

    ax.set_xticks(range(len(neurotransmitters)), labels=neurotransmitters, rotation=-45, ha="right", rotation_mode="anchor")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ax.text(5.7, 0, f'({int(np.sum(counts))})', va='center', ha='left', color='black')

    ax.set_title('Class proportions')
    ax.set_yticklabels([])

    fig.colorbar(img, ax=ax, orientation='horizontal', label='Proportion')

    plt.show()


def confusion_matrix(prediction_list, 
                     epochs, 
                     split):

    confusion_matrix = np.zeros((len(neurotransmitters), len(neurotransmitters)))
    for pred in prediction_list:
        truth = pred[1]
        prediction = pred[0]
        confusion_matrix[truth, prediction] += 1
        
    initial_confusion_matrix = confusion_matrix.copy()
        
    total_list = []
    for row in confusion_matrix:
        total = sum(row)
        total_list.append(row)
        row /= total
    confusion_matrix=100*confusion_matrix

    fig, ax = plt.subplots(figsize=(7,7), dpi=150)

    im = ax.imshow(confusion_matrix, cmap='YlGn')

    ax.set_yticks(range(len(neurotransmitters)), labels=neurotransmitters)
    ax.set_xticks(range(len(neurotransmitters)), labels=neurotransmitters, rotation=-45, ha="right", rotation_mode="anchor")
    ax.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False)
    for i in range(len(neurotransmitters)):
        for j in range(len(neurotransmitters)):
            text = ax.text(j, i, s=f'{round(confusion_matrix[i, j], ndigits=2)}%\n({int(initial_confusion_matrix[i,j])})', ha="center", va="center", color="black")

    for i, row in enumerate(initial_confusion_matrix):
        ax.text(5.75, i, f'({int(sum(row))})',
                va='center', ha='left', color='black')
        
    for j, row in enumerate(initial_confusion_matrix.T):
        ax.text(j, 5.75, f'({int(sum(row))})',
                va='center', ha='center', color='black')

    ax.text(5.75, 5.75, f'({int(np.sum(initial_confusion_matrix))})',
                va='center', ha='left', color='black')

    fig.tight_layout()
    ax.set_title(f'Confusion matrix for filtered class-wise predictions - {split} - Epochs={epochs}')

    #fig.colorbar(im, ax=ax, orientation='vertical', label='Accuracy')

    plt.show()