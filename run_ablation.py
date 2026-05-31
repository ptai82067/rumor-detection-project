"""
Ablation Study Runner — PHEME Rumor Detection
Runs 5 configs and generates figures + tables
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os, sys, csv, warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sentence_transformers import SentenceTransformer

# Style
plt.rcParams.update({'figure.dpi':150,'savefig.dpi':300,'font.size':13,'axes.titlesize':15,
    'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,
    'savefig.facecolor':'white','figure.facecolor':'white','axes.facecolor':'white',
    'savefig.bbox':'tight','savefig.pad_inches':0.2})

DATA = 'data/processed'
OUT = 'results/ablation'

# 1. LOAD DATA
print("Loading data...")
base_df = pd.read_csv(f'{DATA}/pheme_features.csv', dtype={'reply_to':str})
graph_df = pd.read_csv(f'{DATA}/graph_features_v2.csv')
print(f"  Base: {base_df.shape}, Graph: {graph_df.shape}")

# Build thread-level dataframe
thread_data = []
for tid, group in base_df.groupby('thread_id'):
    source = group[group['depth']==0].iloc[0] if len(group[group['depth']==0])>0 else group.iloc[0]
    thread_data.append({
        'thread_id': int(tid), 'label': int(source['label']),
        'text': ' '.join(group['text'].dropna().astype(str).tolist()),
        'source_text': str(source['text'])
    })
thread_df = pd.DataFrame(thread_data)
thread_df = thread_df.merge(graph_df, on='thread_id', how='left')
print(f"  Threads: {thread_df.shape}, Nulls: {thread_df.isnull().sum().sum()}")

# 2. PROPAGATION FEATURES
print("Building propagation features...")
prop_cols = ['thread_size','max_depth','avg_depth_prop','reply_rate']
prop_data = base_df.groupby('thread_id').agg(
    thread_size=('thread_size','first'),
    max_depth=('max_depth','first')).reset_index()
avg_d = base_df.groupby('thread_id')['depth'].mean().reset_index()
avg_d.columns = ['thread_id','avg_depth_prop']
cnt = base_df.groupby('thread_id').size().reset_index()
cnt.columns = ['thread_id','total_posts']
prop_data = prop_data.merge(avg_d, on='thread_id').merge(cnt, on='thread_id')
prop_data['reply_rate'] = prop_data['total_posts'] / (prop_data['thread_size'] + 1)
thread_df = thread_df.merge(prop_data[['thread_id']+prop_cols], on='thread_id', how='left')

# 3. MINILM EMBEDDINGS
print("Generating MiniLM embeddings (this may take a while)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(thread_df['source_text'].fillna('').tolist(), show_progress_bar=True, batch_size=64)
np.save(f'{DATA}/minilm_embeddings_thread.npy', embeddings)
print(f"  Embeddings: {embeddings.shape}")

# 4. SPLIT
y = thread_df['label'].values
train_idx, test_idx = train_test_split(
    np.arange(len(thread_df)), test_size=0.2, random_state=42, stratify=y)
y_train, y_test = y[train_idx], y[test_idx]
print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

# 5. RUN ALL CONFIGS
results = {}
configs = ['TF-IDF','MiniLM','MiniLM+Prop','MiniLM+Graph','Full_Hybrid']
dnames = ['TF-IDF','MiniLM','+Propagation','+Graph(KG v2)','Full Hybrid']
fnames = ['10K','384','384+4','384+14','384+4+14']
graph_cols = ['thread_depth','num_nodes','num_edges','avg_branching_factor',
    'max_branching_factor','source_reply_count','leaf_ratio','avg_depth',
    'source_pagerank','avg_pagerank','source_centrality','avg_centrality',
    'user_rumor_ratio','unique_users']
X_prop = thread_df[prop_cols].fillna(0).values
X_graph = thread_df[graph_cols].fillna(0).values

# Config 1: TF-IDF
print("\nConfig 1: TF-IDF...")
tfidf_vec = TfidfVectorizer(max_features=10000, stop_words='english',
                             ngram_range=(1,2), min_df=2, max_df=0.95)
X_tfidf = tfidf_vec.fit_transform(thread_df['text'].fillna(''))
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf.fit(X_tfidf[train_idx], y_train); yp = clf.predict(X_tfidf[test_idx])
cm = confusion_matrix(y_test, yp)
results['TF-IDF'] = {
    'Acc':accuracy_score(y_test,yp),'Prec':precision_score(y_test,yp,zero_division=0),
    'Recall':recall_score(y_test,yp,zero_division=0),'F1':f1_score(y_test,yp,zero_division=0),
    'FN':int(cm[1][0]),'CM':cm}

# Config 2: MiniLM
print("Config 2: MiniLM...")
X = embeddings
ss = StandardScaler()
X_t = ss.fit_transform(X[train_idx]); X_e = ss.transform(X[test_idx])
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf.fit(X_t, y_train); yp = clf.predict(X_e)
cm = confusion_matrix(y_test, yp)
results['MiniLM'] = {
    'Acc':accuracy_score(y_test,yp),'Prec':precision_score(y_test,yp,zero_division=0),
    'Recall':recall_score(y_test,yp,zero_division=0),'F1':f1_score(y_test,yp,zero_division=0),
    'FN':int(cm[1][0]),'CM':cm}

def run_config(name, X_data):
    ss = StandardScaler()
    X_t = ss.fit_transform(X_data[train_idx]); X_e = ss.transform(X_data[test_idx])
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_t, y_train); yp = clf.predict(X_e)
    cm = confusion_matrix(y_test, yp)
    results[name] = {
        'Acc':accuracy_score(y_test,yp),'Prec':precision_score(y_test,yp,zero_division=0),
        'Recall':recall_score(y_test,yp,zero_division=0),'F1':f1_score(y_test,yp,zero_division=0),
        'FN':int(cm[1][0]),'CM':cm}

# Config 3: MiniLM + Prop
print("Config 3: MiniLM+Prop...")
run_config('MiniLM+Prop', np.hstack([embeddings, X_prop]))

# Config 4: MiniLM + Graph
print("Config 4: MiniLM+Graph...")
run_config('MiniLM+Graph', np.hstack([embeddings, X_graph]))

# Config 5: Full Hybrid
print("Config 5: Full Hybrid...")
run_config('Full_Hybrid', np.hstack([embeddings, X_prop, X_graph]))

# 6. PRINT RESULTS
print("\n" + "="*90)
print("ABLATION STUDY RESULTS")
print("="*90)
print(f"{'Config':<20} {'Feat':<10} {'Acc':<9} {'Prec':<9} {'Recall':<9} {'F1':<9} {'FN':<7}")
print("-"*75)
for i,c in enumerate(configs):
    r = results[c]
    print(f"{dnames[i]:<20} {fnames[i]:<10} {r['Acc']:.4f}    {r['Prec']:.4f}    "
          f"{r['Recall']:.4f}    {r['F1']:.4f}    {r['FN']}")

# 7. SAVE TABLE
os.makedirs(f'{OUT}/figures', exist_ok=True)
base_r = results['TF-IDF']['Recall']
base_fn = results['TF-IDF']['FN']
with open(f'{OUT}/ablation_table.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['Config','Features','Acc','Prec','Recall','F1','FN',
                'Delta_Recall_vs_TFIDF','Delta_FN_vs_TFIDF'])
    for i,c in enumerate(configs):
        r=results[c]; star=' *' if c=='Full_Hybrid' else ''
        w.writerow([dnames[i]+star,fnames[i],round(r['Acc'],4),round(r['Prec'],4),
                    round(r['Recall'],4),round(r['F1'],4),r['FN'],
                    round(r['Recall']-base_r,4),int(base_fn-r['FN'])])
with open(f'{OUT}/ablation_table.md','w',encoding='utf-8') as f:
    f.write('# Ablation Study Results\n\n')
    f.write('| Config | Features | Acc | Prec | Recall | F1 | FN | DRecall vs TF-IDF | DFN vs TF-IDF |\n')
    f.write('|--------|----------|-----|------|--------|----|----|-------------------|---------------|\n')
    for i,c in enumerate(configs):
        r=results[c]
        f.write(f'| {dnames[i]} | {fnames[i]} | {r["Acc"]:.4f} | {r["Prec"]:.4f} | '
                f'{r["Recall"]:.4f} | {r["F1"]:.4f} | {r["FN"]} | '
                f'{r["Recall"]-base_r:+.4f} | {int(base_fn-r["FN"]):+d} |\n')
print(f"  Tables saved to {OUT}/")

# 8. FIGURE 1: Grouped bar chart
print("Generating figures...")
fig, ax = plt.subplots(figsize=(14,7))
metrics = ['Acc','Prec','Recall','F1']
x = np.arange(len(metrics)); width = 0.17
colors = ['#2196F3','#FF5722','#4CAF50','#FF9800','#9C27B0']
for i,c in enumerate(configs):
    vals = [results[c][m] for m in metrics]
    offset = (i-2)*width
    bars = ax.bar(x+offset, vals, width, label=dnames[i], color=colors[i],
                  edgecolor='black', linewidth=0.5)
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for i in range(len(metrics)):
    ax.axvspan(i-width/2-0.01, i+width/2+0.01, alpha=0.05, color='red', zorder=-1)
ax.set_ylabel('Score', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metrics, fontweight='bold')
ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02,1))
ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig1_ablation_bar_metrics.png', dpi=300)
plt.close()
print("  Fig1 saved")

# 9. FIGURE 2: Recall + FN dual-axis
fig, ax1 = plt.subplots(figsize=(12,6))
recalls = [results[c]['Recall'] for c in configs]
fns = [results[c]['FN'] for c in configs]
ax1.plot(dnames, recalls, color='red', marker='o', linewidth=2.5, markersize=8, zorder=5)
ax1.set_ylabel('Recall', color='red', fontweight='bold', fontsize=14)
ax1.tick_params(axis='y', labelcolor='red'); ax1.set_ylim(0.5, 1.0)
for i,(n,v) in enumerate(zip(dnames, recalls)):
    ax1.annotate(f'{v:.4f}', (i,v), textcoords='offset points', xytext=(0,15),
                 ha='center', fontsize=10, fontweight='bold', color='red')
ax2 = ax1.twinx()
bars = ax2.bar(dnames, fns, alpha=0.3, color='gray', width=0.5, zorder=1)
ax2.set_ylabel('False Negatives (FN)', fontweight='bold', fontsize=14)
for bar,val in zip(bars,fns):
    ax2.text(bar.get_x()+bar.get_width()/2., bar.get_height()+5, f'{val}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
fn_red = fns[0]-fns[4]
ax2.annotate(f'Giam {fn_red} FN\nso voi TF-IDF', xy=(4,fns[4]),
             xytext=(2.5,fns[4]+fns[0]//2), fontsize=11, fontweight='bold', color='green',
             arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig2_recall_fn_trend.png', dpi=300)
plt.close()
print("  Fig2 saved")

# 10. FIGURE 3: Confusion matrices
cms = [results[c]['CM'] for c in configs]
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
fig.suptitle('Confusion Matrices - Ablation Study', fontweight='bold', fontsize=16, y=1.05)
for idx, (cn, dn, cm) in enumerate(zip(configs, dnames, cms)):
    ax = axes[idx]
    cm_n = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    ax.imshow(cm_n, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            c = cm[i,j]; p = cm_n[i,j]*100
            col = 'white' if cm_n[i,j] > 0.5 else 'black'
            ax.text(j, i, f'{c:,}\n({p:.1f}%)', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=col)
    ax.add_patch(Rectangle((0,1), 1, 1, fill=False, edgecolor='red', linewidth=2.5, linestyle='--'))
    ax.set_title(f'{dn}\nRecall={results[cn]["Recall"]:.4f}', fontsize=10, fontweight='bold')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Non-Rumor','Rumor'], fontsize=9)
    ax.set_yticklabels(['Non-Rumor','Rumor'], fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig3_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Fig3 saved")

# 11. FIGURE 4: Feature contribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
steps = ['TF-IDF\n-> MiniLM', '+ Propagation', '+ Graph (KG)', 'Full Hybrid']
rec_deltas = [
    results['MiniLM']['Recall'] - results['TF-IDF']['Recall'],
    results['MiniLM+Prop']['Recall'] - results['MiniLM']['Recall'],
    results['MiniLM+Graph']['Recall'] - results['MiniLM']['Recall'],
    results['Full_Hybrid']['Recall'] - results['MiniLM+Prop']['Recall'],
]
fn_deltas = [
    -(results['TF-IDF']['FN'] - results['MiniLM']['FN']),
    -(results['MiniLM']['FN'] - results['MiniLM+Prop']['FN']),
    -(results['MiniLM']['FN'] - results['MiniLM+Graph']['FN']),
    -(results['MiniLM+Prop']['FN'] - results['Full_Hybrid']['FN']),
]
gs = ['#A5D6A7','#66BB6A','#388E3C','#1B5E20']
bars1 = ax1.barh(steps, rec_deltas, color=gs, edgecolor='black', linewidth=0.5)
for bar,val in zip(bars1, rec_deltas):
    lbl = f'+{val:.4f}' if val>=0 else f'{val:.4f}'
    ax1.text(bar.get_width()+0.001 if val>=0 else bar.get_width()-0.005,
             bar.get_y()+bar.get_height()/2., lbl,
             ha='left' if val>=0 else 'right', va='center', fontsize=10, fontweight='bold')
ax1.axvline(x=0, color='black', linewidth=0.8)
ax1.set_xlabel('Delta Recall', fontweight='bold')
ax1.set_title('Improvement in Recall', fontweight='bold', fontsize=13)
bars2 = ax2.barh(steps, fn_deltas, color=['#EF9A9A','#E57373','#D32F2F','#B71C1C'],
                 edgecolor='black', linewidth=0.5)
for bar,val in zip(bars2, fn_deltas):
    lbl = f'{val:+d} cases'
    ax2.text(bar.get_width()+1 if val>=0 else bar.get_width()-2,
             bar.get_y()+bar.get_height()/2., lbl,
             ha='left' if val>=0 else 'right', va='center', fontsize=10, fontweight='bold')
ax2.axvline(x=0, color='black', linewidth=0.8)
ax2.set_xlabel('Delta False Negatives', fontweight='bold')
ax2.set_title('Reduction in FN', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig4_feature_contribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Fig4 saved")

# 12. QUALITY CHECK
print("\n" + "="*50)
print("IMAGE QUALITY CHECK")
print("="*50)
all_ok = True
for fname in ['fig1_ablation_bar_metrics.png','fig2_recall_fn_trend.png',
              'fig3_confusion_matrices.png','fig4_feature_contribution.png']:
    fp = f'{OUT}/figures/{fname}'
    if os.path.exists(fp):
        kb = os.path.getsize(fp)/1024
        s = 'OK' if kb > 150 else 'TOO SMALL'
        if kb <= 150: all_ok = False
        print(f"  [{s}] {fname}: {kb:.0f}KB")
    else:
        print(f"  [MISSING] {fname}"); all_ok = False

# 13. SUMMARY
print("\n" + "="*55)
print("         ABLATION STUDY - KET QUA")
print("="*55)
for c,n in zip(configs,dnames):
    r=results[c]
    print(f"  {n:<20} Acc={r['Acc']:.4f}  Recall={r['Recall']:.4f}  FN={r['FN']}")
print("="*55)
print(f"  Graph features source: graph_features_v2.csv")
print(f"  Figures: {OUT}/figures/")
print(f"  Tables: ablation_table.csv + .md")
print(f"  All images OK: {all_ok}")
print("="*55)
print("\nDONE!")