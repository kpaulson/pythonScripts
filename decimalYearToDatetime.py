import sys

def decimalYearToDatetime(decimalYearInput=2000.139077, verbose=True):
    '''
    This routine is just a simple conversion from a decimal year value to a datetime object. I am trying to write this in a way that it will be easily callable in some other script if I ever decide to grow up and use real programming.
    
    input:
        decimalYearInput:  the rank0 value of the time in decimal year format
        verbose:           True/False selection as to whether it should print out the decimal time and datetime
    
    output:
        datetimeOutput: the Datetime variable time
    '''
    
    from datetime import datetime as dt
    
    year = int(float(decimalYearInput))
    # month = (decimalYearInput - year)*12
    yearFraction = float(decimalYearInput) - int(year)
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)
    secondsInYear = (startOfNextYear - startOfThisYear ) * yearFraction
    
    datetimeOutput = dt(year, month=1, day=1) + secondsInYear
    
    if verbose == True:
        print('input time: ', decimalYearInput)
        print('output time:', datetimeOutput.isoformat(timespec='minutes'), datetimeOutput.strftime("(%j)"))
    
    return(datetimeOutput)
    
if __name__ == '__main__':
    # Map command line arguments to function arguments.
    decimalYearToDatetime(*sys.argv[1:])
    