#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib as mpl
#mpl.use('pgf')
import matplotlib.pyplot as plt
from plottools import figsize, get_colors#, MPL_OPTIONS

MARKER_DICT = {'markersize': 5,
              'linestyle': '',
              'marker': '+',
              'clip_on': True,
              'zorder': 99,
              'color': 'C3'}

L_WIDTH = 3

MPL_OPTIONS = {                         # setup matplotlib to use latex for output
#    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
 #   "text.usetex": True,                # use LaTeX to write all text
 #   "font.family": "serif",
 #   "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
 #   "font.sans-serif": [],
#    "font.monospace": [],
    "font.size": 16,
    "axes.labelsize": 16,               # LaTeX default is 10pt font.
    "legend.fontsize": 16,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.top': False,
    'ytick.right': False,
    #'axes.prop_cycle': mpl.cycler('color', get_colors('meu')),
 #   "pgf.preamble": [
 #       r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
 #       r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
 #       r"\usepackage{amsmath}",
 #       r"\usepackage{amsfonts}",
#        r"\usepackage{nicefrac}",
#        r"\newcommand{\vect}[1]{\boldsymbol #1}",
  #      ]
    }

#plt.style.use('default')
mpl.rcParams.update(MPL_OPTIONS)

import numpy as np
#from matplotlib import rc
#rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


def local_model_gen(model_range, Theta, C, N, sort_idx, poly):
    Theta = Theta[sort_idx, :]
    C = C[:, sort_idx]
    if N == 1:
        for m, m_range in enumerate(np.array(model_range)[sort_idx]):
            u = np.linspace(*m_range[0])
            y = Theta[m, :] @ poly.fit_transform(u.reshape(-1, 1)).T
            c = Theta[m, :] @ poly.fit_transform(C[:, m].reshape(-1, 1)).T
            yield u, y, c
    else:
        print("Local models only available for N == 1")


class PlotterLLM:
    def __init__(self, options, sort=False):
        self.options = options
        n = len(options)
        self.fig, self.axes = plt.subplots(nrows=n, ncols=1, figsize=figsize(1.0, ratio=1.15))
        self.axes = np.array(self.axes).flatten()
        self.sort = sort
        self.standard_colors = plt.get_cmap('plasma')(np.linspace(0, 0.7, 4))
    
    # --- plot options 2D ---
    
    def result(self, ax):
        ax.set_title("Lernergebnis", fontsize=16)
        #ax.grid(True)
        ax.plot(self.u, self.y, color=self.standard_colors[0], label="$y_{true}$", linewidth=L_WIDTH)
        ax.plot(self.u, self.y_pred, color=self.standard_colors[2], label="$y_{approx}$", linewidth=L_WIDTH)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        
        #half_err = (self.y_pred - self.y)
        #upper = (self.y_pred + half_err).ravel()
        #lower = (self.y_pred - half_err).ravel()
        #ax.fill_between(self.u.ravel(), upper, lower,
        #                color=self.standard_colors[2], alpha=0.2)
        ax.plot(self.X_train, self.y_train, '+', color=self.standard_colors[1],
                markersize=12, alpha=0.75, label='Daten')
        ax.legend()

    def error(self, ax):
        ax.grid(True)
        ax.plot(self.u, self.y_pred - self.y, color=self.standard_colors[0])
        ax.get_yaxis().get_major_formatter().set_powerlimits((0, 0))
        ax.set_ylabel("Absoluter Fehler")
        ax.get_yaxis().get_offset_text().set_x(-0.05)
        ax.get_yaxis().get_offset_text().set_y(0.98)
    
    def validity_function(self, ax):
        ax.set_title("Zugehoerigkeitsfunktionen", fontsize=16)
        ax.set_ylabel(r"$\tilde{\Phi}_i$")
        A = np.zeros((self.llm.M, self.u.shape[0]))
        self.llm.network.validity_function(self.u, A)
        for idx, column in enumerate(A[self.sort_idx, :]):
            ax.plot(self.u, column, color=self.colors[idx], linewidth=L_WIDTH)
            if self.llm.M <= 20:
                ax.annotate(f"{self.sort_idx[idx]}", xy=(self.llm.C[:, self.sort_idx][:, idx], np.max(column)),
                            xytext=(-4, -20), textcoords='offset points', fontsize=12)
            
    def models(self, ax):
        ax.set_title("Lokale Modelle", fontsize=16)
        #ax.grid(True)
        ax.set_ylabel(r"$\hat{y}_i$")
        #ax.plot(self.X_train, self.y_train, 'x', color=self.standard_colors[1],
        #        markersize=5, alpha=0.5)#, label="$y_{train}$")
        
        #ax2 = ax.twinx()
        for idx, (u, y, c) in enumerate(local_model_gen(self.llm.network.model_range, self.llm.network.local_models.Theta_,
                                                        self.llm.network.C, self.llm.network.N_, self.sort_idx,
                                                        self.llm.network.local_models.poly)):
            
            # bar plot with samples per model
            """
            number_of_samples = np.sum(self.llm.network.A[self.sort_idx, :][idx, :])
            ax2.bar(self.llm.C[:, self.sort_idx][:, idx], number_of_samples, width=5*number_of_samples/self.X_train.shape[0], alpha=0.25, color=self.colors[idx])
            """

            ax.plot(u, y, color=self.colors[idx], linewidth=L_WIDTH)
            #markerline, stemlines, _ = ax.stem(self.llm.C[:, self.sort_idx][:, idx], c, '--')
            #plt.setp(markerline, color=self.colors[idx], alpha=0.5)
            #plt.setp(stemlines, color=self.colors[idx], alpha=0.5)

            if self.llm.M <= 20:
                ax.annotate(f"{self.sort_idx[idx]}", xy=(self.llm.C[:, self.sort_idx][:, idx], c),
                            xytext=(-4, 5), textcoords='offset points', fontsize=12)
            
        #ax.legend()
        
    def report(self, ax):
        
        # linear regression
        grad = 2 * self.llm.training_duration * 1000 / (self.llm.M + 1)**2
                
        m, s = divmod(self.llm.training_duration, 60)
        #ax.set_title(f"Bericht\nTrainingszeit: {int(m)}:{s:.2f} | "
        #             + "$rmse_{min}$=" + f"{np.min(self.llm.global_loss):.4e}"
        #             + r" | $\bar{t}_{grad} =$" + f"{grad:.4f}")
        ax.grid(True)
        ax.plot(range(1, self.llm.M + 1), self.llm.global_loss, color=self.standard_colors[0])
        ax.set_ylabel("Trainingsfehler (RMSE)")
        ax.set_xlabel("Anzahl Teilmodelle")
        ax.semilogy(True)
        ax.set_xlim(left=1)
        tol_line = lines.Line2D([1, self.llm.M + 1], [self.llm.training_tol, self.llm.training_tol], alpha=0.75,
                                lw=1, color='red', axes=ax, linestyle='dashed')
        ax.add_line(tol_line)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
        #ax_ = ax.twinx()
        #d = np.array(self.llm.network.split_duration)*1000
        #ax_.plot(range(1, self.llm.M), d, color=self.standard_colors[1], alpha=0.5)
        #ax_.set_ylabel('Dauer pro Split in $ms$', color=self.standard_colors[1])
        #ax_.tick_params('y', colors=self.standard_colors[1])

    def volume(self, ax):
        volume = []
        mr = self.llm.network.model_range
        for r in mr:
            ext = np.prod([np.abs(np.subtract(*r_n)) for r_n in r])
            volume.append(ext)

        idx = np.argsort(volume)
        ax.plot(self.llm.network._get_local_loss(self.X_train, self.y_train)[idx], alpha=0.75)
        ax.set_ylabel("Local Loss")
        ax.set_xlabel("Modell")
        ax.grid(True)

        ax2 = ax.twinx()
        ax2.step(range(1, self.llm.M + 1), np.array(volume)[idx], 'y')
        ax2.semilogy(True)
        ax2.set_ylabel("Ausdehnung der Modelle")
        ax2.tick_params('y', colors='y')

    def distribution(self, ax):
        try:
            x_grid = np.linspace(self.X_train.min(), self.X_train.max(), 1000)
            ax.plot(x_grid, np.exp(self.llm.kde_.score_samples(x_grid[:, np.newaxis])))
            ax.set_xlabel("Eingangsraum")
            ax.set_ylabel("geschätze WDF")
            ax.grid(True)

            ax2 = ax.twinx()
            weights = np.diagonal(self.llm.network.R_.toarray())
            ax2.scatter(self.X_train, weights, color='y')
            ax2.set_ylabel("Gewichtung")
            ax2.tick_params('y', colors='y')
        except AttributeError as e:
            print("[WARNING]: Plotting distribution failed.")
            print(e)

    # --- plot options 3D ---
    
    def result3D(self, ax):
        ax.set_title("Lernergebnis")
        ax.grid(True)     
        c = ax.contour(*self.u, self.y, 10, colors='black')
        c.collections[-1].set_label("true")
        cs = ax.contour(*self.u, self.y_pred, 10, cmap='viridis')
        norm = Normalize(vmin=np.min(self.y_pred), vmax=np.max(self.y_pred))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        self.fig.colorbar(sm, ax=ax)
        ax.legend()
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    def error3D(self, ax):
        ax.set_title("Absoluter Fehler")
        ax.grid(True)
        c = ax.contourf(*self.u, self.y_pred - self.y, 50, cmap='coolwarm')
        self.fig.colorbar(c, ax=ax)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    
    def validity_function3D(self, ax, A):
        ax.set_title("Zugehörigkeitsfunktionen")
        for idx, column in enumerate(A):
            # ax.plot(self.u, column, color=self.colors[idx])
            k = self.u[0].shape[0]
            cs = ax.contour(*self.u, np.reshape(column, (k, k)), 3, cmap='viridis')
        
        norm = Normalize(vmin=np.min(column), vmax=np.max(column))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        sm.set_array([])
        self.fig.colorbar(sm, ax=ax)
        ax.scatter(self.llm.C[:, self.sort_idx][0, :], self.llm.C[:, self.sort_idx][1, :],
                   marker='x', color=self.standard_colors[0])#, label=r"$\c$")
        #ax.legend(loc=4, bbox_to_anchor=(0., 1.0, 1., .102))
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        
        
    def models3D(self, ax):
        ax.set_title("Teilmodelle")
        for idx, r in enumerate(self.llm.network.model_range):
            r = list(zip(*r))
            ax.add_patch(patches.Rectangle(xy=r[0], width=r[1][0]-r[0][0], height=r[1][1]-r[0][1], fill=False)) 
        ax.scatter(self.llm.network.C[:, self.sort_idx][0, :], self.llm.network.C[:, self.sort_idx][1, :],
                   marker='x', color=self.standard_colors[0])#, label=r"$\c$")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        #ax.legend(loc=4, bbox_to_anchor=(0., 1.0, 1., .102))
    
    def report3D(self, ax):
        self.report(ax)         
        
    # ----------------
    
    def plot(self):
        abc = list(map(chr, range(65, 91)))
        self.fig.set_size_inches((15, len(self.options)*2.8))
        for idx, option in enumerate(self.options):
            self.axes[idx].cla()
            #self.axes[idx].set_title(abc[idx], weight='bold', loc='left')
            if self.N < 2:
                getattr(self, option)(self.axes[idx])
            else:
                getattr(self, option + "3D")(self.axes[idx])
        #self.fig.tight_layout()
    
    def update(self, u, y, X_train, y_train, llm):
        self.u = u
        self.y = y
        self.X_train, self.X_val, self.y_train, self.y_val = llm._make_validation_split(X_train, y_train)
        self.llm = llm
        
        self.N = 1 if not isinstance(u, list) else len(u)
        
        if self.N == 1:
            self.y_pred = np.reshape(llm.predict(u), (len(u), 1))
        elif self.N == 2:
            X = np.vstack((u_i.flatten() for u_i in u)).T
            k = u[0].shape[0]
            self.y_pred = np.reshape(llm.predict(X), (k, k))
            self.y = np.reshape(y, (k, k))
        else:
            self.options = ["report"]

        # define colors for each model
        self.colors = plt.get_cmap('inferno')(np.linspace(0, 0.7, llm.M))
        
        if self.sort and u.shape[1] <= 1:
            self.sort_idx = np.argsort(llm.C, axis=1)[0]
        else:
            self.sort_idx = range(llm.C.shape[1])
            
        self.plot()