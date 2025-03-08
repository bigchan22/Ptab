# import optax
import numpy as np
# from Load_Data import batch
from src.data import batch_e


def train(loss_val_gr, opt_update, params, opt_state, features, rows, cols, ys, masks):
    curr_loss, gradient = loss_val_gr(params, features, rows, cols, ys, masks)
    updates, opt_state = opt_update(gradient, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, curr_loss


def compute_accuracies(model, params_to_evaluate, dataset, batch_size=100):
    total_correct = 0.0
    for i in range(0, len(dataset.features), batch_size):
        b_features, b_rows, b_cols, b_ys, b_edges = batch_e(
            dataset.features[i:i + batch_size], dataset.rows[i:i + batch_size],
            dataset.columns[i:i + batch_size], dataset.labels[i:i + batch_size],
            dataset.edge_types[i:i + batch_size])

        accs = model.accuracy(params_to_evaluate, b_features, b_rows, b_cols, b_ys,
                              b_edges)
        total_correct += accs * len(dataset.features[i:i + batch_size])
    return total_correct / len(dataset.features)


def print_accuracies(model, params_to_evaluate,
                     test_dataset,
                     train_dataset,
                     batch_size=100):
    train_accuracy = compute_accuracies(
        model, params_to_evaluate, dataset=train_dataset, batch_size=batch_size)
    test_accuracy = compute_accuracies(
        model, params_to_evaluate, dataset=test_dataset, batch_size=batch_size)

    combined_accuracy = np.average(
        [train_accuracy, test_accuracy],
        weights=[len(train_dataset.features),
                 len(test_dataset.features)])
    print(f'Train accuracy: {train_accuracy:.3f} | '
          f'Test accuracy: {test_accuracy:.3f} | '
          f'Combined accuracy: {combined_accuracy:.3f}')
    return test_accuracy
