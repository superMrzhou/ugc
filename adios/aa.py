from functools import wraps
class Logit(object):
    def __init__(self, logfile='out.log'):
        self.logfile = logfile
        print self.logfile

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kargs):
            log_string = func.__name__ + ' is called'
            # self.logfile = 'out.log'
            with open(self.logfile, 'a') as f:
                f.writelines(log_string)
            self.notify()  # do something
            return func(*args, **kargs)

        return wrapper

    def notify(self):
        print('notify')

class email_logit(Logit):
    def __init__(self, email='admin@qq.com', *args, **kargs):
        self.email = email
        # super(Logit,self).__init__(*args, **kargs)
        Logit.__init__(self,*args, **kargs)

    def notify(self):  # not only log,also email
        print('send a email')


@email_logit(email='admin@qq.com',logfile='out.log')
def myfunc():
    print('in email decorator')


myfunc()
