class FallbackError(RuntimeError):
    code = 'FALLBACK_ERROR'

class InvalidRequest(FallbackError):
    code = 'INVALID_REQUEST'

class UnsupportedDomain(FallbackError):
    code = 'UNSUPPORTED_DOMAIN'

class MissingDefinition(FallbackError):
    code = 'MISSING_SOURCE_DEFINITION'
    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result

class SourceArithmeticError(FallbackError):
    code = 'SOURCE_ARITHMETIC_ERROR'

class SourceFieldOverflow(FallbackError):
    code = 'SOURCE_FIELD_OVERFLOW'
