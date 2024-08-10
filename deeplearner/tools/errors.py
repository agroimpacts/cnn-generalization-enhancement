class InputError(Exception):
    '''
    Exception raised for errors in the input
    '''

    def __init__(self, message):
        '''
        Params:
            message (str): explanation of the error

        '''

        self.message = message

    def __str__(self):
        '''
        Define message to return when error is raised
        '''

        if self.message:
            return 'InputError, {} '.format(self.message)
        else:
            return 'InputError'
