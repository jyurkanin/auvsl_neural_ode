#pragma once

#include "types/System.h"

#include <memory>

template<typename Scalar>
class SystemFactory
{
public:
	virtual std::shared_ptr<System<Scalar>> makeSystem() = 0;
};
