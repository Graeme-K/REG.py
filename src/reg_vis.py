import numpy as np
import pandas as pd
from scipy.stats import norm # type: ignore
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)

def plot_violin(file_list=[], save = False, file_name='reg_violin.png'):
    labels = [i+1 for i in range(len(file_list))]
    fig=plt.figure()
    fig.suptitle(r'Distribution of $R$ in each segment')
    ax = fig.add_subplot(111)
    ax.violinplot(file_list, showmeans='True')
    set_axis_style(ax, labels)
    ax.set_xlabel('Segments')
 
    if save == True:
        fig.savefig(file_name, dpi=300)

    return ax


def generate_data_vis(file, file_list, n_term, save = False, file_name='data_vis.png', title = "REG data visualization" ):

    (mu, sigma) = norm.fit(file['R'])
    fig = plt.figure(constrained_layout=False, figsize=[10,8])
    fig.suptitle(title)       
    gs1 = fig.add_gridspec(nrows=6, ncols=6, top=0.9, bottom=0.65, hspace=0, wspace = 0.40)
    ax1 = fig.add_subplot(gs1[0:6,0:2]) # violin
    ax2 = fig.add_subplot(gs1[0:3,2:6]) # hist
    ax3 = fig.add_subplot(gs1[3:6,2:6]) #steam
    gs2 = fig.add_gridspec(nrows=6, ncols=6, top=0.55, bottom=0.05, hspace=0.05)
    ax4 = fig.add_subplot(gs2[:,:]) # data

##Violin_plot
    labels = [i+1 for i in range(len(file_list))]
    parts = ax1.violinplot(file_list, showmeans='True')
    set_axis_style(ax1, labels)
    for i in range(len(parts['bodies'])):
        if file.R.equals(file_list[i]) == True:
            parts['bodies'][i].set_color('green')
            parts['bodies'][i].set_facecolor('green')
            parts['bodies'][i].set_edgecolor('green')
    ax1.set_xlabel('Segments')
    ax1.grid(axis='y', alpha=0.75)
    ax1.set_title(r'Distribution of $R$')



##histogram plot
    ax2.hist(file['R'], bins=[0.1*i for i in range(-11,11,1)],
             density=0 , color = 'green', edgecolor='black', linewidth=1.2, alpha=0.4)
    ax2.grid(axis='y', alpha=0.75)
    ax2.grid(axis='x', alpha=0.75)
    ax2.set_xlim(-1,1)
    ax2.xaxis.set_ticks(np.arange(-1.0, 1.1, 0.1))
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')
    ax2.set_ylabel('Count Number')
    ax2.set_xlabel(r'$R$')
    ax2.yaxis.set_ticks_position('right')
    #ax2.axvline(mu, ls='--', color='r')

##stem plot1
    ax3.set_xlim(-1, 1)
    ax3.stem(file['R'], file['REG'],'k', markerfmt=' ', use_line_collection ='True')
    ax3.grid(axis='x', alpha=0.75)
    ax3.set_ylabel('REG value')
    ax3.xaxis.set_ticks(np.arange(-1.0, 1.1, 0.1))
    ax3.yaxis.set_ticks([])
    ax3.set_xlabel(r'$R$')
    #ax3.axvline(mu, ls='--', color='r')
##Stem plot2
    pos = pd.DataFrame(columns=file.columns)
    neg = pd.DataFrame(columns=file.columns)    
    cond = file.REG < 0
    rows = file.loc[cond,:]
    neg = neg.append(rows, ignore_index=True)    
    cond = file.REG > 0
    rows = file.loc[cond,:]
    pos = pos.append(rows, ignore_index=True)        
    markerline, stemlines, baseline = ax4.stem(pos['R'], pos['REG'], 'b', use_line_collection ='True')
    markerline.set_markerfacecolor('b')
    markerline.set_markeredgecolor('b')    
    markerline2, stemlines2, baseline2 = ax4.stem(neg['R'], neg['REG'], 'r', use_line_collection ='True')
    markerline2.set_markerfacecolor('r')
    markerline2.set_markeredgecolor('r')    
    ax4.set_xlim(-1, 1)
    bbox_props = dict(boxstyle="round,pad=0.1", fc="w", ec="k", lw=0, alpha=0.8)
    cond = pos.R > 0
    temp = pos.loc[cond,:]
    n = temp.nlargest(n_term, 'REG').reset_index(drop=True)  
    text_p = [ax4.text(n['R'][i], n['REG'][i], n['TERM'][i], bbox=bbox_props) for i in range(len(n))]

    cond = neg.R < 0
    temp = neg.loc[cond,:]
    n = temp.nsmallest(n_term, 'REG').reset_index(drop=True)  
    text_n = [ax4.text(n['R'][i], n['REG'][i], n['TERM'][i], bbox=bbox_props) for i in range(len(n))]

    ax4.set_title('Relevant IQA contributions')
    ax4.set_xlabel(r'$R$')
    ax4.set_ylabel('REG value')
    
    if save == True:
        fig.savefig(file_name, dpi=300)
    
    return

def plot_segment(coordinate, wfn_energy, critical_points, label=False, color=True, annotate=True,
                 title='REG segments', y_label='Energy', x_label='Coordinate', save =False, file_name='segments.png'):
        
    color_list=['blue', 'darkgreen', 'darkred', 'yellow', 'cyan', 'magenta', 'grey', 'salmon',
                'seagreen', 'aquamarine', 'lightgreen', 'silver', 'lime', 'indigo', 'indianred']

    fig = plt.figure(figsize=(12,5))
    graph = plt.subplot(111)
    graph.plot(coordinate,wfn_energy,'o', color='black')
    if label == True:
        for i in range(len(wfn_energy)):
            graph.annotate(str(i+1), [coordinate[i], wfn_energy[i]])

    
    graph.xaxis.set_major_locator(MaxNLocator(nbins=10))
    graph.xaxis.set_minor_locator(AutoMinorLocator())    
    graph.set_title(title, fontsize =16)  
    graph.set_xlim([graph.get_xlim()[0], graph.get_xlim()[1]])
    graph.set_ylabel(y_label, fontsize =14)
    graph.set_xlabel(x_label, fontsize=14)
    graph.axvline(coordinate[0], linestyle = 'dotted', color = 'black') #the first segment starts at the global minimum.
    
    for i in critical_points: #split the segments
        graph.axvline(coordinate[i], linestyle = 'dotted', color = 'black')
          
    if color == True: #Color_segments
        graph.axvspan(graph.get_xlim()[0], graph.get_xlim()[1], facecolor = color_list[len(critical_points)], alpha = 0.2)
        start = 0
        for i in range(len(critical_points)):
            if coordinate[start] > coordinate[critical_points[i]]:
                graph.axvspan(coordinate[critical_points[i]], coordinate[start], facecolor='white', alpha =1)
                graph.axvspan(coordinate[critical_points[i]], coordinate[start], facecolor=color_list[i], alpha =0.2)
            start=critical_points[i]            
        start = 0   
        for i in range(len(critical_points)):
            if coordinate[start] < coordinate[critical_points[i]]:
                graph.axvspan(coordinate[start], coordinate[critical_points[i]], facecolor='white', alpha =1)
                graph.axvspan(coordinate[start], coordinate[critical_points[i]], facecolor=color_list[i], alpha =0.2)
            start=critical_points[i]
                
    if annotate== True: #Write segment number:
        text = []
        start=0
        y_pos = graph.get_ylim()[1] 
        x_pos = coordinate[start]
        text.append(graph.text(x_pos,  y_pos, '1',fontsize=14))
        
        for i in range(len(critical_points)):
            start=critical_points[i]
            x_pos = coordinate[start]
            text.append(graph.text(x_pos, y_pos,  str(i+2), fontsize=14))
            start=critical_points[i]
    fig.autofmt_xdate()
    if save == True:
        fig.savefig(file_name, dpi=300)

    return
    
# Rungs used to auto-calibrate the significance threshold, as fractions of the
# largest |REG| in the table.  Walked coarse to fine, so a system whose leading
# terms are tightly bunched stops high up and a spread-out one walks further down.
REG_FRACTION_LADDER = (1/2, 1/3, 1/4, 1/5, 1/8, 1/10, 1/15, 1/20, 1/30, 1/50, 1/100)


def select_significant_terms(dataframe, min_rows=10, max_rows=25, r_threshold=0.0,
                             degeneracy_tol=0.02):
    """Return the significant terms of a REG table, with the cut chosen automatically.

    This replaces the older "top n_terms positive plus top n_terms negative"
    selection.  A single magnitude threshold is applied to both signs, so the
    positive/negative split is whatever the physics gives — a segment driven by two
    negative terms and eight positive ones is reported that way instead of being
    padded to 4 and 4.

    The threshold is a fraction of the largest |REG| in this table, which makes it
    scale-free: it does not care whether the leading term is 16 kJ/mol or 0.5.  The
    fraction is not fixed, because a fixed one (max/10, say) gives a handful of rows
    on one system and hundreds on the next.  Instead REG_FRACTION_LADDER is walked
    from coarse to fine and the first rung that admits at least `min_rows` terms is
    taken.  If that rung overshoots `max_rows` — a table with no real separation,
    such as dispersion, where everything sits within a factor of two of the top — the
    ladder is abandoned and the table is truncated by rank at `max_rows` instead.

    Either way the cut is then extended downwards through any terms within
    `degeneracy_tol` of the last one kept, so near-degenerate contributions are not
    split by an arbitrary boundary.

    `r_threshold` is a weak sanity guard, not the decider: it removes terms whose
    correlation with the control coordinate is so poor that their REG value is
    meaningless.  On a well-behaved segment almost every term has |R| > 0.95, so it
    normally removes nothing.  Set it to 0 to disable.

    The chosen threshold and fraction are recorded on the returned frame's ``attrs``
    (``reg_threshold``, ``reg_fraction``) so they can be quoted in a caption.
    """
    col = ['TERM', 'REG', 'R'] if 'TERM' in dataframe.columns else list(dataframe.columns)
    df = dataframe.loc[:, col].dropna(axis=0, how='any', subset=['REG', 'R'])
    if r_threshold:
        df = df[df['R'].abs() >= r_threshold]
    df = df[df['REG'].abs() > 0]
    if len(df) == 0:
        empty = df.reset_index(drop=True)
        empty.attrs['reg_threshold'], empty.attrs['reg_fraction'] = None, None
        return empty

    df = df.reindex(df['REG'].abs().sort_values(ascending=False).index)
    magnitude = df['REG'].abs().values
    largest = magnitude[0]

    fraction, n_keep = None, None
    for f in REG_FRACTION_LADDER:
        count = int((magnitude >= f * largest).sum())
        if count >= min_rows:
            fraction, n_keep = f, count
            break
    if n_keep is None:  # even the finest rung stays under min_rows — keep what there is
        fraction = REG_FRACTION_LADDER[-1]
        n_keep = max(int((magnitude >= fraction * largest).sum()), min(min_rows, len(df)))
    if n_keep > max_rows:  # no separation to find; fall back to a plain rank cut
        fraction, n_keep = None, max_rows
    else:
        # Do not split near-degenerate terms across the boundary.  Only worth doing
        # when the ladder found a real cut: in the rank-cut fallback the values form
        # a continuum and this would walk on indefinitely, so max_rows stays a cap.
        while (n_keep < len(magnitude) and n_keep < max_rows
               and magnitude[n_keep] >= magnitude[n_keep - 1] * (1 - degeneracy_tol)):
            n_keep += 1

    out = df.iloc[:n_keep].sort_values('REG').reset_index(drop=True)
    out.attrs['reg_threshold'] = float(magnitude[n_keep - 1])
    out.attrs['reg_fraction'] = fraction
    return out


def pandas_REG_dataframe_to_table(dataframe, table_name, SAVE_FIG=True):
    if SAVE_FIG==True:
        if len(dataframe) == 0 or len(dataframe.columns) == 0:
            return  # nothing passed the significance filter — no table to draw
        dataframe['R'] = np.round(dataframe['R'], decimals=3)
        dataframe['REG'] = np.round(dataframe['REG'], decimals=2)
        # segments no longer contribute an equal number of rows, so the shorter
        # ones are padded with NaN by the side-by-side concatenation; blank those
        # cells rather than printing "nan" in the table.
        cell_text = dataframe.astype(object).where(dataframe.notna(), '').values
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.axis('tight')
        t= ax.table(cellText=cell_text, colWidths = [0.4]*len(dataframe.columns),  colLabels=dataframe.columns,  cellLoc='center',loc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(12)
        fig.savefig(table_name, dpi=300, bbox_inches="tight")

def create_term_dataframe(reg_dataframe, headers, i):
    temp = [reg_dataframe[0][i], reg_dataframe[1][i]]
    df = pd.DataFrame(temp).transpose()
    df.columns = ["REG", "R"]
    df.index = headers
    df = df.rename_axis('TERM').reset_index()
    return df

def filter_term_dataframe(prop_dataframe, original_prop_name, new_prop_name):
    col = ['TERM', 'REG', 'R']
    mask = prop_dataframe['TERM'].str.contains(original_prop_name, regex=False)
    new_df = prop_dataframe.loc[mask, col].copy()
    new_df = new_df.sort_values('REG').reset_index(drop=True)

    def format_term(term):
        label = term
        if term.startswith(original_prop_name + '-'):
            label = term[len(original_prop_name) + 1:]
        elif term.startswith(original_prop_name + '_'):
            label = term[len(original_prop_name) + 1:]
        return f"{new_prop_name}({label.replace('_', ',')})"

    new_df['TERM'] = new_df['TERM'].map(format_term)
    return new_df