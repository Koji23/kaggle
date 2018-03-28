export PATH="${PATH}:/usr/local/anaconda/bin"
export PS1="\[\033[38;5;6m\]\h\[$(tput sgr0)\] \[\033[38;5;15m\]\w\[$(tput sgr0)\] \[\033[38;5;7m\]#\[$(tput sgr0)\] "

alias ls='ls --color'
alias d='ls -Alh --group-directories-first'
alias cx='chmod +x'
alias x='exit'
alias fastai='cd /usr/local/fastai && source activate fastai-cpu && ipython'
alias notebook='jupyter notebook --allow-root --ip=0.0.0.0 --notebook-dir=/usr/local/notebooks'
