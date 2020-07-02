import sys
import logging as syslog

log = syslog.getLogger('bo')


def set_log(level=syslog.INFO):
    ch = syslog.StreamHandler(sys.stdout)
    ch.setLevel(level)
    fmt = syslog.Formatter('%(message)s')
    ch.setFormatter(fmt)
    log.addHandler(ch)
    log.setLevel(level)

    def wrap(orig):
        def new_func(*args, **kwargs):
            try:
                left = ' '.join(str(x) for x in args)
                right = ' '.join(f'{k}={v}' for k, v in kwargs.items())
                new = ' '.join(filter(None, [left, right]))
                import arrow
                now = arrow.now()
                time = now.strftime('%H:%M:%S') + '.{:03d}'.format(now.time().microsecond // 1000)
                import inspect
                stack = inspect.stack()[1]
                import os
                file = os.path.basename(stack.filename).replace('.py', '')
                new = f'[{time}][qb][{file:>13}:{stack.lineno:>3}] {new}'
            except:
                new = f'{args} {kwargs}'
            orig(new)

        return new_func

    log.debug = wrap(log.debug)
    log.info = wrap(log.info)
    log.warning = wrap(log.warning)
    log.exception = wrap(log.exception)

