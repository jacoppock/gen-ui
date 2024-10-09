import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import React from 'react';

interface FHIRContextProps {
    content?: string;
    isLoading?: boolean;
}

// Generic Resource Display
const GenericResourceDisplay: React.FC<{ resource: any }> = ({ resource }) => (
    <>
        <CardTitle className="flex items-center gap-2">
            {resource.resourceType}
            <Badge variant="outline">{resource.id}</Badge>
        </CardTitle>
        <CardDescription>
            <ScrollArea className="h-[200px] w-[400px] rounded-md border p-4">
                {Object.entries(resource).map(([key, value]) => (
                    <div key={key} className="mb-2">
                        <span className="font-semibold">{key}:</span> {JSON.stringify(value)}
                    </div>
                ))}
            </ScrollArea>
        </CardDescription>
    </>
);

// Patient Resource Display
const PatientDisplay: React.FC<{ resource: any }> = ({ resource }) => (
    <>
        <CardTitle className="flex items-center gap-2">
            {resource.name?.[0]?.given.join(' ')} {resource.name?.[0]?.family}
            <Badge variant="outline">{resource.gender}</Badge>
        </CardTitle>
        <CardDescription>
            <div>Birth Date: {resource.birthDate}</div>
            <div>ID: {resource.id}</div>
        </CardDescription>
    </>
);

// Observation Resource Display
const ObservationDisplay: React.FC<{ resource: any }> = ({ resource }) => (
    <>
        <CardTitle>{resource.code?.text || 'Observation'}</CardTitle>
        <CardDescription>
            <div>Value: {resource.valueQuantity?.value} {resource.valueQuantity?.unit}</div>
            <div>Date: {resource.effectiveDateTime}</div>
        </CardDescription>
    </>
);

// Add more resource-specific displays here...

const resourceDisplays: { [key: string]: React.FC<{ resource: any }> } = {
    Patient: PatientDisplay,
    Observation: ObservationDisplay,
    // Add more resource types here...
};

export function ContentLoading(): JSX.Element {
    return (
        <Card className="w-[450px]">
            <CardHeader className="grid grid-cols-[1fr] items-start gap-4 space-y-0">
                <div className="space-y-1">
                    <CardTitle>
                        <Skeleton className="h-[18px] w-[100px]" />
                    </CardTitle>
                    <CardDescription>
                        <div className="flex flex-col gap-[2px] pt-[4px]">
                            {Array.from({ length: 5 }).map((_, i) => (
                                <Skeleton
                                    key={`description-${i}`}
                                    className="h-[12px] w-[150px]"
                                />
                            ))}
                        </div>
                    </CardDescription>
                </div>
            </CardHeader>
            <CardContent>
                <Skeleton className="h-[12px] w-full" />
            </CardContent>
        </Card>
    );
}

export function FHIRContext({ content, isLoading = false }: FHIRContextProps): JSX.Element {
    if (isLoading) {
        return <ContentLoading />;
    }

    if (!content) {
        return (
            <Card className="w-[450px]">
                <CardHeader className="items-start space-y-1">
                    <CardTitle>No Data</CardTitle>
                    <CardDescription>No FHIR resource data available.</CardDescription>
                </CardHeader>
            </Card>
        );
    }

    let parsedContent;
    try {
        parsedContent = JSON.parse(content.replace(/'/g, '"'));
    } catch (error) {
        console.error("Error parsing JSON:", error);
        return (
            <Card className="w-[450px]">
                <CardHeader className="items-start space-y-1">
                    <CardTitle>Error</CardTitle>
                    <CardDescription>Unable to parse FHIR data.</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="text-sm text-muted-foreground">{content}</div>
                </CardContent>
            </Card>
        );
    }

    const ResourceDisplay = resourceDisplays[parsedContent.resourceType] || GenericResourceDisplay;

    return (
        <Card className="w-[450px]">
            <CardHeader className="items-start space-y-1">
                <ResourceDisplay resource={parsedContent} />
            </CardHeader>
            <CardContent>
                <div className="text-sm text-muted-foreground">
                    Resource Type: {parsedContent.resourceType}
                </div>
            </CardContent>
        </Card>
    );
}

export default FHIRContext;