import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import SpectralBiclustering
from scipy.stats import chi2_contingency, ranksums
from statsmodels.stats.multitest import fdrcorrection
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


from utils import constants, load_classification_data, get_simple_pipeline

def run_hierarchical_clustering():
    inputs, labels = load_classification_data(delta=0.5, is_sigma=True)
    inputs = inputs.drop(columns=[
        'pmcid', 'link', 'NCT Number', 'result', 'result_translation_success',
        'clinical_treatment_name', 'preclinical_treatment_name',
        'clinical_dosage_frequency', 'preclinical_dosage_frequency',
        'preclinical_year'  # exclude these columns
    ], errors='ignore')

    # list categorical features here
    cat_features = [
        'animal_strain',
        'animal_species',
        'animal_sex',
        'preclinical_dosage_amount_unit',
        'preclinical_administration_route',
        'disease_induction_method',
        'age_groups',
        'gender',
        'phases',
    ]

    # Convert categorical columns to string explicitly
    for col in cat_features:
        inputs[col] = inputs[col].astype(str)


    # Ensure only existing categorical features are used
    cat_features = [col for col in constants.FEATURES_CAT if col in inputs.columns]
    # cat_features = inputs.select_dtypes(include=['object', 'category']).columns.tolist()


    inputs_transformed = get_simple_pipeline(
        inputs,
        cat_features,
        scaling_method_name='minmax',
    ).fit_transform(inputs)

    inputs_transformed = pd.DataFrame(inputs_transformed).reset_index(drop=True)
    labels = labels.reset_index(drop=True).rename('label')
    data = pd.concat([inputs_transformed, labels], axis=1)

    color_mapper = {
        1: sns.color_palette()[0],
        0: sns.color_palette()[1],
    }

    sns.clustermap(
        data.drop(columns=['label']),
        metric='euclidean',
        row_colors=data['label'].map(color_mapper),
        cmap='Blues',
        figsize=(10, 20),
    )
    plt.savefig("hc_all.png")
    plt.close()

    for val in [0, 1]:
        subset = data.query(f'label == {val}')
        sns.clustermap(
            subset.drop(columns=['label']),
            metric='euclidean',
            row_colors=subset['label'].map(color_mapper),
            cmap='Blues',
            figsize=(10, 20),
        )
        plt.savefig(f"hc_{val}.png")
        plt.close()



def run_biclustering():
    # === Load the TSV file ===
    df = pd.read_csv("../../data/cleaned_t2dm_a2h_05_21.tsv", sep='\t')

    # === Define the label ===
    label_col = 'result_translation_success'
    labels = df[label_col].astype(int)

    # === Drop non-feature columns ===
    non_feature_cols = [
        'pmcid', 'link', 'NCT Number', 'result', label_col,
        'clinical_treatment_name', 'preclinical_treatment_name',
        'preclinical_year'  
    ]
    feature_df = df.drop(columns=non_feature_cols)

    # === Define categorical and numerical columns ===
    categorical_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

    # === Preprocessing pipeline ===
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
            ]), numerical_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))  # compatible with all versions
            ]), categorical_cols)
        ]
    )

    preprocessor.set_output(transform="default")

    # === Apply transformation ===
    X_array = preprocessor.fit_transform(feature_df)

    # Convert to dense if sparse
    if hasattr(X_array, "toarray"):
        X_array = X_array.toarray()

    # Create feature names
    cat_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
    all_feature_names = numerical_cols + list(cat_feature_names)

    # Create DataFrame with feature names
    X = pd.DataFrame(X_array, columns=all_feature_names)

    # === Attach labels back for plotting ===
    X.insert(0, "Translation success", labels.values)
    data = X.copy()
    data = data.reset_index(drop=True)

    # === Spectral Biclustering ===
    N_ROW_CLUSTERS = 8
    N_COL_CLUSTERS = 32

    model = SpectralBiclustering(
        n_clusters=(N_ROW_CLUSTERS, N_COL_CLUSTERS),
        method='scale',
        random_state=0
    )
    model.fit(X.drop(columns=["Translation success"]))

    # === Reorder data ===
    data_reordered = data.iloc[:, model.column_labels_.argsort() + 1]  # +1 to skip label column
    data_reordered.insert(0, "Translation success", labels.values)
    data_reordered = data_reordered.iloc[model.row_labels_.argsort()]

    # print all column names
    # print(data_reordered.columns[1:].tolist())
    # Print all column names that start with "preclinical_dosage_frequency"
    # matching_columns = [col for col in data_reordered.columns if col.startswith("clinical_dosage_frequency")]
    # print(matching_columns)

    feature_name_map = pd.read_csv("feature_name_map.csv")
    feature_name_map = dict(
        zip(feature_name_map['feature_name_original'], feature_name_map['feature_name'])
    )
    data_reordered.columns = [
        feature_name_map[x] if x in feature_name_map else x
        for x in list(data_reordered.columns)
    ]

    # === Plot ===
    color_mapper = {1: sns.color_palette()[0], 0: sns.color_palette()[1]}

    g = sns.clustermap(
        data_reordered.drop(columns=['Translation success']),
        metric='euclidean',
        row_cluster=False,
        col_cluster=False,
        row_colors=data_reordered['Translation success'].map(color_mapper),
        cmap='Blues',
        figsize=(20, 25)
    )


    # Move colorbar to upper-left of heatmap
    g.cax.set_position([0.15, 0.7, 0.025, 0.075])  
    # [left, bottom, width, height] in figure coordinates

    g.cax.set_title("Feature\nValue\n(Minmax)", fontsize=14, pad=12)
    

    # Add bicluster boundaries
    for i in range(N_ROW_CLUSTERS - 1):
        g.ax_heatmap.axhline(
            (model.row_labels_ <= i).sum(),
            color='red',
            linewidth=1,
            linestyle='--'
        )
    for i in range(N_COL_CLUSTERS - 1):
        g.ax_heatmap.axvline(
            (model.column_labels_ <= i).sum(),
            color='red',
            linewidth=1,
            linestyle='--'
        )

    # X-axis labels: Feature names
    g.ax_heatmap.set_xticks(np.arange(len(data_reordered.columns) - 1))  # exclude label
    # Set ticks at the center of each column
    num_features = len(data_reordered.columns) - 1
    g.ax_heatmap.set_xticks(np.arange(num_features) + 0.5)
    g.ax_heatmap.set_xticklabels(data_reordered.columns[1:], rotation=90)

    g.ax_heatmap.set_xlabel("Feature", fontsize=24)
    g.ax_heatmap.set_ylabel("Preclinical-clinical pair", fontsize=24)
    g.ax_heatmap.set_yticks([])

    # Legend
    # handles = [Patch(facecolor=color_mapper[label]) for label in [1, 0]]
    # plt.legend(
    #     handles,
    #     ['Yes', 'No'],
    #     title='Translation\nSuccess',
    #     bbox_to_anchor=(1, 1),
    #     bbox_transform=plt.gcf().transFigure,
    #     loc='upper right'
    # )
    # Legend for labels.
    handles = [Patch(facecolor=color_mapper[label]) for label in [1, 0]]

    legend = g.ax_heatmap.legend(
        handles,
        ['Yes', 'No'],
        title='Translation\nSuccess',
        loc='lower right',
        bbox_to_anchor=(-0.038, -0.0054),  # Move left of the heatmap
        fontsize=12,                 # Make labels larger
        title_fontsize=14            # Make title larger
    )
    legend.get_title().set_ha('center')



    # Save plot
    plt.savefig("biclustering.png", bbox_inches='tight')
    plt.close()







    # Just plot the x axis labels.
    data_reordered_small = data_reordered.iloc[:2]
    g = sns.clustermap(
        data_reordered_small.drop(columns=['Translation success']),
        metric='euclidean',
        row_cluster=False,
        col_cluster=False,
        row_colors=data_reordered_small['Translation success'].map(color_mapper),
        cmap='Blues',
        figsize=(20, 10),
    )
    # Legend for labels.
    handles = [
        Patch(facecolor=color_mapper[label]) for label in [1, 0]
    ]
    plt.legend(
        handles,
        ['Yes', 'No'],
        title='Translation\nSuccess',
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc='upper right',
    )


    plt.savefig("biclustering_xticklabels.svg")
    plt.close()


    data_dump = data_reordered.copy()
    data_dump['row_label'] = np.sort(model.row_labels_)
    data_dump = data_dump.set_index(['row_label', data_dump.index])
    columns = pd.MultiIndex.from_arrays(
        [
            list(np.sort(model.column_labels_)) + [''],
            data_dump.columns
        ],
        names=['col_label', 'feature'],
    )
    data_dump.columns = columns
    print(data_dump)
    data_dump.to_csv("bc_data.csv")










def run_biclustering_analysis():
    data = pd.read_csv(
        "bc_data.csv",
        skiprows=[2],
        index_col=[0, 1],
        header=[1],
    )

    # Best cluster.
    result_rows = []
    for feature in data.columns[:-1]:
        value_counts = data[feature].value_counts()
        if len(value_counts) == 1:
            continue
        else:
            # Select here which clusters to analysis in the bc_results file
            for i in [0, 1, 5]:
                result = {
                    'cluster': i,
                    'feature': feature,
                }
                data_cluster = data.loc[i]
                for j in data.index.get_level_values(0).unique():
                    if i == j:
                        result['method'] = 'self'
                        result[f'cluster_{j}_p'] = np.nan
                        continue

                    data_other = data.loc[j]
                    if len(value_counts) == 2:
                        result['method'] = 'chi2'
                        neg_cluster = len(data_cluster.query(f"`{feature}` == 0.0"))
                        pos_cluster = len(data_cluster.query(f"`{feature}` == 1.0"))
                        neg_other = len(data_other.query(f"`{feature}` == 0.0"))
                        pos_other = len(data_other.query(f"`{feature}` == 1.0"))
                        try:
                            _, p, _, expected = chi2_contingency([
                                [neg_cluster, pos_cluster],
                                [neg_other, pos_other],
                            ])
                            if (expected < 5).any():
                                result[f'cluster_{j}_p'] = np.nan
                            else:
                                result[f'cluster_{j}_p'] = p
                        except Exception:
                            result[f'cluster_{j}_p'] = np.nan

                    else:
                        # Numerical, so use ranksum.
                        result['method'] = 'ranksum'
                        _, p = ranksums(
                            data_cluster[feature],
                            data_other[feature],
                        )
                        result[f'cluster_{j}_p'] = p
                result_rows += [result]
    result = pd.DataFrame(result_rows)
    result = result.sort_values(['cluster'])

    # FDR.
    p_vals_df = result[result.columns[-8:]]
    p_vals = p_vals_df.values.flatten()
    p_vals = p_vals[~np.isnan(p_vals)]
    p_vals = list(fdrcorrection(p_vals, alpha=0.01)[1])

    p_vals_np = p_vals_df.copy().values
    for i in range(p_vals_np.shape[0]):
        for j in range(p_vals_np.shape[1]):
            if np.isnan(p_vals_np[i, j]):
                continue
            p_vals_np[i, j] = p_vals.pop(0)
    p_vals_df = pd.DataFrame(
        p_vals_np,
        columns=p_vals_df.columns,
        index=p_vals_df.index,
    )
    result[result.columns[-8:]] = p_vals_df

    result['significant'] = result[result.columns[-8:]].apply(
        lambda row: False if row.notna().sum() != 7 else row.max() < 0.01,
        axis=1,
    )
    result.to_csv("bc_result.csv")


if __name__ == '__main__':
    run_hierarchical_clustering()
    run_biclustering()
    run_biclustering_analysis()