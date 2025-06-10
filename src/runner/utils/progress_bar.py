import sys
def progress_bar(
                iteration, 
                total_iterations, 
                epoch= None, 
                total_epochs= None, 
                vars= None,
                prefix='',
                postfix='',
                style='bar',
                bar_width=10):
        """
        Display a live training progress bar in the console.
        Args:
            epoch (int): Current epoch number.
            total_epochs (int): Total number of epochs.
            loss (float, optional): Training loss.
            val_loss (float, optional): Validation loss.
            acc (float, optional): Accuracy.
            fold (int, optional): Current fold number for K-Fold Cross-Validation.
            bar_width (int, optional): Width of the progress bar.
        """
        progress = iteration / total_iterations
        filled_len = int(progress * bar_width)
        progress_vis = None
        if style == 'bar':
            progress_vis = "█" * filled_len + '-' * (bar_width - filled_len)
        elif style == 'arrow':
            progress_vis = ">" * filled_len + '-' * (bar_width - filled_len)
        elif style == 'dots':
            progress_vis = "•" * filled_len + '-' * (bar_width - filled_len)
        
        s = ''
        if prefix: 
            s += prefix + ' | '
        if epoch is not None:
            width = len(str(total_epochs))
            s += f"Epoch {epoch:0{width}d}/{total_epochs} | "

        if iteration is not None and total_iterations is not None:
            width = len(str(total_iterations))
            s += f"Iter {iteration:0{width}d}/{total_iterations} | "
        
        s += f"[{progress_vis}] | "
        s += postfix + ' | '

        if vars is not None:
            for key, value in vars.items():
                if value is not None:
                    if isinstance(value, int):
                        s += f"{key}: {value} | "
                    elif isinstance(value, float):
                        if value < 1e-2:
                            s += f"{key}: {value:.4e} | "
                        else:
                            if value < 1:
                                s += f"{key}: {value:.4f} | "
                            else:
                                s += f"{key}: {value:.4f} | "
                    else:
                        s += f"{key}: {value} | "
        s = s.rstrip(' | ')
        
        sys.stdout.write('\r' + s)
        sys.stdout.flush()